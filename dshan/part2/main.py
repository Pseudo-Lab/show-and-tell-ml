import gradio as gr
import os
import json
import openai
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
import umap
import plotly.express as px
import hdbscan
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import pandas as pd

key = os.environ.get('OPENAI_API_KEY')
client = openai.OpenAI(api_key=key)

"""
# 토픽 기반 문서 필터링
LLM -> 대용량 문서 처리 시 속도 및 비용 문제
상대적으로 작은 임베딩 모델을 활용하여 전체 문서의 주제를 자동으로 분류 -> 작업자가 탐색해야 할 문서의 범위 축소, 효율적 자원 사용
"""
def run_topic_modeling(files):
    docs = pd.read_excel(files)
    docs['contents'] = [str(line).strip() for line in docs['판례내용']]
    docs['timestamp'] = [datetime.strptime(line, '%Y-%m-%d').strftime('%Y%m%d') for line in docs['선고일자']]

    random_state = 1800

    legal_stopwords = list(set(["계약", "소송", "법원", "법률", "법적", "조항", "법", "의무", "권리"
                      ,"공소외", "사건", "피고인", "피고가","피고는", "피고에게","피고의","피고인은","피고인의","피고인이", "청구인"
                            ,"청구외", "피청구인", "사건", "정보", "피고", "피고들의","원고들의","원고", "원고인","원고가","원고에게","원고는","원고의","원고와","참가인",
                            "피청구인의","피심판청구인",
                           "같은","있다","상대방","없다","관한","대한","간의","별지","있는","파기하고","대하여","같이",
                            "동의를","등을","동의","기준으로","기준","또는","한다","이를","선고","등의","청구인의","제1항","확인안됨",
                           "사건을","따라","원심은","원심판결","위와","관하여","원심판결을","것으로","상고를","위법이",
                           "의하면", "하여", "의하여", "원심이", "판결", "청구인과",
                           "비추어", "원을", "거기에", "공동피고인", "것이다", "피청구인이",
                           "위하여", "사실을", "원심의", "검사의","처분의", "청구인이","청구인을","청구인은","청구인에게",
                            "사건본인들의","기각한다","내용의","기재","1의",
                           "민법", "가입대상", "소속기관의", "조치를", "피해자의", "대법원", 
                            "상고이유", "가입신청을", "가입의사를", "민법", "피고들은","상고비용은"]))
    
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    umap_model = umap.UMAP(random_state=random_state)
    hdbscan_model = hdbscan.HDBSCAN(prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words=legal_stopwords)
    representation_model = MaximalMarginalRelevance(diversity=0.2)
    topic_model = BERTopic(verbose=True,
                              language='korean',
                              umap_model=umap_model,
                              hdbscan_model=hdbscan_model,
                              embedding_model=embedding_model,
                              vectorizer_model=vectorizer_model,
                              representation_model=representation_model,
                              calculate_probabilities=True)

    corpus_embedding = embedding_model.encode(docs['contents'].tolist(),convert_to_tensor=True)
    corpus_embedding = corpus_embedding.cpu()
    topics, probs = topic_model.fit_transform(docs['contents'].tolist(), corpus_embedding.numpy())

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info.Topic != -1]  # Remove outliers

    df = pd.DataFrame({
        "문서": list(range(len(docs['contents']))),
        "본문": docs['contents'],
        "토픽": topics
    })
    return topic_info, df.to_json()

"""
# 선택적 문서 요약 및 도움 이미지 제공
multimodal LLM -> 제공한 조건을 바탕으로 문서 요약 태스크를 수행 + 프롬프트 기반의 이미지 생성
               -> 작업 시작점을 앞당길 수 있어 시간, 비용적 절감 / 전문가 평가를 플로우에 도입하여 피드백 루프 형성 가능(HITL?)
"""
def summarize_and_visualize(topic_number, json_df):
    global client
    
    df = pd.read_json(json_df)
    selected_docs = df[df["토픽"] == topic_number]

    if selected_docs.empty:
        return "선택된 토픽에 해당하는 문서가 없습니다.", None
        
    
    ##########################토픽에서 처음 등장하는 문서를 대표 문서로 판단##################################
    print(selected_docs.iloc[0]["본문"])
    content = selected_docs.iloc[0]["본문"][:4000]
    ##########################토픽에서 처음 등장하는 문서를 대표 문서로 판단##################################
    
    response = client.responses.create(
      model="gpt-4o-mini",
      input=[
        {
          "role": "system",
          "content": [
            {
              "type": "input_text",
              "text": f"의도를 파악할 수 있도록, 구체적으로 설명하여 용어 자체보다 개념과 상황을 이해할 수 있도록 해석하여 초등학생 2-3학년도 아는 단어로 쓰여진 쉬운 글로 바꿔줘. 아래의 형식을 지켜줘.\n    1. 알아두면 좋은 단어\\n\n    2. 전체 내용 요약\\n\n    3. 등장인물\\n\n    4. 내용\\n\n    5. 법원의 결정\\n \n    6. 정리\\n {content}"
            }
          ]
        }
      ],
      text={
        "format": {
          "type": "text"
        }
      },
      reasoning={},
      tools=[],
      temperature=1,
      max_output_tokens=2048,
      top_p=1,
      store=True
    )
    print(response)
    summary = response.output_text
    
    image_prompt = f"다음 내용을 표현할 수 있는 4컷만화를 그려줘:\n\n{summary}"
    
    ##########################현재 공개된 모델의 질적 한계 ##################################
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        n=1,
        size="1024x1024"
    )
    ##########################현재 공개된 모델의 질적 한계 ##################################
    image_url = image_response.data[0].url
    print(image_url)

    return summary, image_url

def preview_topic_documents(topic_num, topic_json):
    df = pd.read_json(topic_json)
    topic_num = int(topic_num)
    preview_df = df[df["토픽"] == topic_num]
    
    ##########################프리뷰의 개선이 필요##################################
    preview_df['본문'] = preview_df['본문'].apply(
        lambda x: "..." + x[len(x)//2 - 75 : len(x)//2 + 75] + "..." if len(x) > 150 else x
    )
    ##########################프리뷰의 개선이 필요##################################
    return preview_df

with gr.Blocks() as demo:
    gr.Markdown("# 💡Easy-read Helper")

    with gr.Row():
        file_input = gr.File(file_types=[".xlsx", ".csv"], file_count="single", label="문서 업로드")
        topic_output = gr.Dataframe(headers=["Topic", "Count", "Name"], label="토픽 결과")
    
    hidden_df = gr.Textbox(visible=False)
    run_button = gr.Button("🗂️ 토픽 모델링 실행")
    run_button.click(run_topic_modeling, inputs=[file_input], outputs=[topic_output, hidden_df])

    with gr.Row():
        topic_num_input = gr.Number(label="선택할 토픽 번호", precision=0)
        preview_output = gr.Dataframe(label="📝 해당 토픽의 문서 미리보기")
    
        topic_num_input.change(fn=preview_topic_documents, inputs=[topic_num_input, hidden_df], outputs=[preview_output])

    with gr.Row():     
        summary_output = gr.Textbox(label="요약된 본문", lines=10, show_copy_button=True)
        image_output = gr.Image(label="생성된 이미지")
    
    summarize_button = gr.Button("🧠 요약 및 이미지 생성")
    summarize_button.click(fn=summarize_and_visualize, inputs=[topic_num_input, hidden_df], outputs=[summary_output, image_output])
    

demo.launch()

"""
개선점)
adapter처럼 foundation model을 복잡한 설정이나 모델 구조에 대한 깊은 이해 없이도 손쉽게 선택할 수 있게 함
workflow에 전문가의 평가를 반영

"""
