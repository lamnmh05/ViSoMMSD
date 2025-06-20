import os
from google import genai
from google.genai import types
import mimetypes # Thêm thư viện mimetypes để đoán MIME type

def generate(image_path, caption_text): # Thêm tham số cho đường dẫn ảnh và caption
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("Lỗi: Biến môi trường GEMINI_API_KEY chưa được thiết lập.")
        return

    client = genai.Client(
        api_key=api_key,
    )

    # Sử dụng model Flash.
    # 'gemini-1.5-flash-latest' là một lựa chọn tốt và phổ biến.
    # Nếu 'gemini-2.0-flash-001' là một model hợp lệ và bạn muốn dùng, hãy thay thế ở đây.
    # model_name = "gemini-2.0-flash-001" # Model bạn yêu cầu
    model_name = "gemini-1.5-flash-latest" # Model Flash phổ biến
    print(f"Sử dụng model: {model_name}")


    try:
        # 1. Đọc ảnh từ file local
        print(f"Đang đọc ảnh từ: {image_path}")
        if not os.path.exists(image_path):
            print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'")
            return

        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        print(f"Đọc ảnh thành công, kích thước: {len(image_bytes)} bytes.")

        # 2. Xác định MIME type từ tên file
        actual_mime_type, _ = mimetypes.guess_type(image_path)
        if not actual_mime_type:
            # Mặc định nếu không đoán được (ví dụ: file không có phần mở rộng)
            # Hoặc bạn có thể báo lỗi và yêu cầu người dùng cung cấp
            print(f"Không thể đoán MIME type cho '{image_path}'. Mặc định là 'application/octet-stream'.")
            actual_mime_type = 'application/octet-stream'
        print(f"MIME type được xác định: {actual_mime_type}")

    except FileNotFoundError: # Đã kiểm tra ở trên, nhưng để đây cho chắc chắn
        print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'")
        return
    except IOError as e:
        print(f"Lỗi IO khi đọc file ảnh '{image_path}': {e}")
        return
    except Exception as e_img_proc:
        print(f"Lỗi không xác định khi xử lý ảnh: {e_img_proc}")
        return

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type=actual_mime_type,
                    data=image_bytes,
                ),
                types.Part.from_text(text=f"Use both the image and caption to classify whether the image and caption is sarcasm or non-sarcasm. Caption: {caption_text}"),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=types.Part.from_text(text="You are an expert in sarcasm detection"),
    )

    try:
        print(f"Đang gửi yêu cầu tới model: {model_name}...")
        # Sử dụng tên model dạng string cho tham số model
        model_instance = client.get_model(f"models/{model_name}") # Lấy model instance

        response_stream = model_instance.generate_content( # Gọi trên model instance
            contents=contents,
            generation_config=generate_content_config,
            stream=True # Để nhận stream
        )
        print("Đã nhận phản hồi từ model, đang xử lý...")
        for chunk in response_stream:
            print(chunk.text, end="")
        print() # Thêm dòng mới sau khi stream kết thúc
    except Exception as e_gen:
        print(f"Lỗi trong quá trình tạo nội dung: {e_gen}")
        # print(f"Model: {model_name}")
        # print(f"Contents parts types: {[type(p) for p in contents[0].parts]}")
        # print(f"Config: {generate_content_config}")

if __name__ == "__main__":

    local_image_path = r"D:\Git_repo\ViSoMMSD\data\all\images_data1\fb_1004.jpg" 
    example_caption = "Chán" 


    generate(local_image_path, example_caption)