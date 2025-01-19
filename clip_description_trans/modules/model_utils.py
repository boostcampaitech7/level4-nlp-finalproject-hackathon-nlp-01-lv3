from transformers import AutoModel, AutoTokenizer

def initialize_model(model_path='OpenGVLab/VideoChat-Flash-Qwen2-7B_res224', mm_llm_compress=False):
    """
    Initializes the model and tokenizer with given configurations.

    Args:
        model_path (str): The path to the model.
        mm_llm_compress (bool): Whether to enable LLM compression.

    Returns:
        tuple: The initialized model, tokenizer, and image processor.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    image_processor = model.get_vision_tower().image_processor

    #Configure the model
    mm_llm_compress = False
    if mm_llm_compress:
        model.config.mm_llm_compress = True
        model.config.llm_compress_type = "uniform0_attention"
        model.config.llm_compress_layer_list = [4, 18]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
    else:
        model.config.mm_llm_compress = True

    return model, tokenizer, image_processor
