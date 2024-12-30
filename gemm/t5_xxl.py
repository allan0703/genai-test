# pip install accelerate

from transformers import T5Tokenizer, T5EncoderModel
import torch

# 检查设备是否支持 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 T5 模型和 Tokenizer
model_name = "city96/t5-v1_1-xxl-encoder-bf16"  # 或者使用 "google/t5-v1_1-xxl" 等模型
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

# 输入文本
text = [
    "a futuristic or sci-fi armored suit, resembling a robot or mech, glows with an intense, pulsing light. The suit's predominantly gray and gold color scheme is now illuminated from within, casting an otherworldly glow. The red glowing eyes seem to burn brighter, as if fueled by the suit's newfound energy. The camera slowly zooms out, revealing a misty, fog-shrouded environment that adds to the sense of mystery and foreboding. The suit's imposing presence and advanced technology are amplified by the eerie atmosphere, exuding an aura of power and menace.",
    "blurry, deformed, disfigured, low quality, software, text, collage, grainy, logo, no visual content, blurred effect, striped background, abstract, illustration, computer generated, distorted"
]

# 将文本编码为张量
inputs = tokenizer(text, return_tensors="pt", padding= "max_length", max_length = 256, truncation=True).to(device)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
with torch.no_grad():
    for i in range(12):
        outputs = model(**inputs)
start_event.record()
# 前向传播，获取编码器输出
with torch.no_grad():
    for i in range(15):
        outputs = model(**inputs)
end_event.record()
torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(end_event)
avg_time_ms = elapsed_time_ms / 15
# 获取最后一层的隐藏状态
# 输出维度: [batch_size, sequence_length, hidden_size]
hidden_states = outputs.last_hidden_state

# 打印结果
print(f"Shape of hidden states: {hidden_states.shape}")
print(f"Hidden states for first token in first sentence: {hidden_states[0, 0, :5]}")
print(f"average time of encoder is {avg_time_ms} ms")
