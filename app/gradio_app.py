import gradio as gr
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter some text.", {}
    
    result = predict(text)
    sentiment = result['sentiment'].upper()
    confidence = result['confidence'] * 100
    scores = result['scores']
    
    emoji = {"POSITIVE": "😊 POSITIVE", "NEGATIVE": "😞 NEGATIVE"}
    label = f"{emoji.get(sentiment, sentiment)} — {confidence:.1f}% confident"
    
    return label, scores

examples = [
    ["Super da machan, vera level!"],
    ["Worst movie ever, total waste of time"],
    ["Romba nalla iruku bro"],
    ["Bore adichuchu, time waste"],
    ["Semma padam da thalaivaaa!"],
    ["Enna da ithu mokka padam"],
    ["Mass entry scene superb ah iruku"],
    ["Pavam story, olunga edukala"],
]

with gr.Blocks(title="Tamil Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌟 Tamil Sentiment Analysis
    ### Lightweight Tamil/Tanglish sentiment classifier — built for real-world deployment
    
    Fine-tuned **MuRIL** (Google) on 55,000+ Tamil-English comments. 
    Achieves **94.7% accuracy** on binary sentiment classification.
    
    > Built as a lightweight alternative to large LLMs for Tamil NLP use cases — 
    > local government portals, small business apps, low-bandwidth mobile apps.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter Tamil/Tanglish text",
                placeholder="Type something like: Super da machan, vera level!",
                lines=3
            )
            submit_btn = gr.Button("Analyze Sentiment 🔍", variant="primary")
            gr.Examples(examples=examples, inputs=text_input)
        
        with gr.Column(scale=1):
            sentiment_output = gr.Textbox(label="Result")
            scores_output = gr.Label(label="Confidence Scores")
    
    gr.Markdown("""
    ---
    **Model:** google/muril-base-cased fine-tuned on DravidianCodeMix dataset  
    **Dataset:** 55,064 balanced Tamil-English sentences (FIRE 2020 + Zenodo)  
    **Accuracy:** 94.7% | **F1 Score:** 94.7%
    """)
    
    submit_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, scores_output]
    )

if __name__ == "__main__":
    demo.launch()