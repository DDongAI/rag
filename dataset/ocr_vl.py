"""
 * @author: zkyuan
 * @date: 2025/4/8 10:53
 * @description: 使用大模型做ocr识别
"""

import base64  # base64 用于处理 Base64 编码
import requests  # requests 用于发送 HTTP 请求
import streamlit as st


# 这段代码定义了一个名为 vision_page 的函数，并设置了页面标题和描述，解释了 GPT-4o 的功能及其当前的限制。
def vision_page():
    st.title("OCR识别")
    if "base_url" not in st.session_state:
        # st.session_state['base_url'] = os.getenv('OPENAI_BASE_URL')
        st.session_state['base_url'] = "http://175.6.13.6:50008/v1"

    if "api_key" not in st.session_state:
        # st.session_state['api_key'] = os.getenv('OPENAI_API_KEY')
        st.session_state['api_key'] = "token-abc123"
    # 初始化参数
    api_key = (
        st.session_state.api_key
        if "api_key" in st.session_state and st.session_state.api_key != ""
        else None
    )
    if api_key is None:
        st.error("Please enter your API key in the home.")
        st.stop()

    if "base_url" in st.session_state:
        base_url = st.session_state.base_url
    else:
        base_url = "https://api.openai.com/v1"

    # 创建一个文件上传器，允许用户上传图片文件，并设置最大文件大小为 5MB。
    upload_images = st.file_uploader("上传文件", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    # 创建一个数字输入框，让用户输入最大 tokens 数量，默认值为 300。
    max_tokens = st.number_input("Max tokens(如果图片内容过多，可以适当调大；如果图片内容少，可以适当调小)", min_value=1, value=300, step=1)

    # 检查上传的图片文件是否超过最大大小，如果没有超过，则读取文件内容并显示图片。
    bytes_data = None
    if upload_images is not None:
        if upload_images.size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            # image = Image.open(upload_images)
            bytes_data = upload_images.getvalue()
            st.image(bytes_data, caption=upload_images.name, width=200)

    # 处理用户输入的提示信息 prompt 和上传的图片
    if prompt := st.chat_input():
        # 如果用户输入了提示信息，则显示用户消息。
        st.chat_message("user").write(prompt)
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                try:
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                    # 如果上传了图片，则将图片转换为 Base64 编码，并构建包含文本和图片的请求负载。
                    if bytes_data is not None:
                        base64_image = base64.b64encode(bytes_data).decode("utf-8")
                        payload = {
                            # "model": "gpt-4o",
                            "model": "Qwen2_5-VL-7B",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": prompt,
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                        },
                                    ],
                                },
                            ],
                            "max_tokens": max_tokens,
                        }
                    else:
                        # 如果没有上传图片，则构建仅包含文本的请求负载。
                        payload = {
                            # "model": "gpt-4o",
                            "model": "Qwen2_5-VL-7B",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt,
                                },
                            ],
                            "max_tokens": max_tokens,
                        }
                    # 发送请求到 OpenAI API
                    if base_url.endswith('/'):
                        base_url = base_url[:-1]
                    response = requests.post(
                        base_url + "/chat/completions", headers=headers, json=payload
                    )

                    # 检查状态码是否正常，不正常会触发异常
                    response.raise_for_status()
                    print(response.json())
                    result = response.json()["choices"][0]["message"]["content"]
                    st.markdown(result)
                except Exception as e:
                    st.error(e)
                    st.stop()


if __name__ == "__main__":
    vision_page()
