from model_handler import initialize_model, patch_cuda_to_cpu,patch_huggingface_cached_module

# Function for text-based chat
def chat_with_text(model, tokenizer, generation_config, question, history=None):
    """
    Handles text-based chat with the model.
    """
    response, history = model.chat(
        tokenizer=tokenizer,
        image_files=None,  # No images for text input
        question=question,
        generation_config=generation_config,
        history=history,
        return_history=True,
    )
    return response, history

# Function for image-based chat
def chat_with_image(model, tokenizer, generation_config, image_file_path, question, history=None):
    """
    Handles image-based chat with the model.
    """
    response, history = model.chat(
        tokenizer=tokenizer,
        image_files=image_file_path,  # Pass the image binary data
        question=question,
        generation_config=generation_config,
        history=history,
        return_history=True,
    )
    return response, history

# Main function to test both functionalities
if __name__ == "__main__":
    # Initialize model, tokenizer, and config
    # Apply the patch

    model, tokenizer, generation_config = initialize_model()

    patch_huggingface_cached_module()

    # Patch the model for CPU usage
    patch_cuda_to_cpu(model)

    # Test text-based chat
    question_text = """What is the basics of how to use the model?"""
    response_text, _ = chat_with_text(model, tokenizer, generation_config, question_text)
    print(f"Text Question: {question_text}\nResponse: {response_text}")

    # Test image-based chat
    #image_file_path = "sample.webp"
    img_cfg_map = {
        'recipt': {'path': './images/recipt.jpeg', 'question': '<image>\nRead the text and provide word by word ocr for the document.'},
        'salad_dressing': {'path': './images/salad_dressing.jpeg', 'question': '<image>\nRead the text in the image'},
        'empty_bag': {'path': './images/empty_bag.jpeg', 'question': '<image>\nRead the text in the image'}
    }
    image_file_path = './images/salad_dressing.jpeg'
    question_image = """<image>\n
    Read the text in the image
    """

    for k, conf in img_cfg_map.items():
        print(f'------------- processing: {k} -------------')
        image_file_path = conf['path']
        question_image = conf['question']
        response_image, _ = chat_with_image(model, tokenizer, generation_config, image_file_path, question_image)
        print(f"Image Question: {question_image}\nResponse: {response_image}")
