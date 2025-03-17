#!/bin/bash

# 命令行模式封装脚本
set -e

# 设备自动检测函数
detect_device() {
  if [[ "$1" == "auto" ]]; then
    if command -v mps &> /dev/null; then
      echo "mps"
    else
      echo "cpu"
    fi
  else
    echo "$1"
  fi
}

# 添加输入文件存在性检查
check_file() {
  if [ ! -f "$1" ]; then
    echo "错误：文件 $1 不存在"
    exit 1
  fi
}

# 统一使用绝对路径
data_dir=$(realpath "$5")
check_file "${data_dir}/$6"
check_file "${data_dir}/$7"

case "$1" in
  train)
    device=$(detect_device "${2:-auto}")
    data_dir=$(realpath "${5:-data/train}")
    check_file "${data_dir}/${6:-en.txt}"
    check_file "${data_dir}/${7:-zh.txt}"

    python3 train.py \
      --batch_size "${3:-32}" \
      --epochs "${4:-10}" \
      --save_dir "$(realpath "${8:-saved_models}")" \
      --data_dir "${data_dir}" \
      --src_file "$6" \
      --tgt_file "$7"
    ;;  
  infer)
    device=$(detect_device "${3:-auto}")
    python3 ../inference.py \
      --model_path "$2" \
      --input_file "${4:-data/test/en.txt}" \

    ;;
  
  finetune)
    device=$(detect_device "${3:-auto}")
    python3 ../finetune.py \
      --model_path "$2" \
      --dataset "${4:-data/finetune}" \

    ;;
  
  *)
    echo "使用方式: $0 {train|infer|finetune} [参数]"
    echo "示例: "
    echo "  $0 train mps 64 20 data/train en.txt zh.txt    # 训练"
    echo "  $0 infer model.pth    # 推理"
    echo "  $0 finetune model.pth # 微调"
    exit 1
    ;;
esac