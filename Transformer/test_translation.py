import torch
from inference import TranslationInference

def test_translation():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化翻译器
    translator = TranslationInference(
        model_path='saved_models/model_best.pth',
        src_vocab_path='data/src_vocab.json',
        tgt_vocab_path='data/tgt_vocab.json',
        device=device
    )
    
    # 测试文本
    test_texts = [
        "Hello world",
        "How are you?",
        "This is a test"
    ]
    
    print("\n测试单个文本翻译:")
    translation = translator.translate(test_texts[0])
    print(f"输入: {test_texts[0]}")
    print(f"输出: {translation}")
    
    print("\n测试批量文本翻译:")
    translations = translator.translate_batch(test_texts)
    for src, tgt in zip(test_texts, translations):
        print(f"输入: {src}")
        print(f"输出: {tgt}\n")

if __name__ == '__main__':
    test_translation() 