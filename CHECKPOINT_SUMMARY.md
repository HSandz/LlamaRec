## Vấn Đề Chính

1. **Resume không hoạt động** - `trainer.train()` không nhận checkpoint path
2. **LoRA weights bị mất** - Luôn khởi tạo mới thay vì load từ checkpoint
3. **Thiếu validation** - Không kiểm tra checkpoint tồn tại

## Đã Sửa

**train_ranker.py:**
- ✅ Validate checkpoint path
- ✅ Load LoRA weights từ checkpoint nếu tồn tại
- ✅ Truyền checkpoint vào `trainer.train()`

**trainer/llm.py:**
- ✅ Xóa `resume_from_checkpoint` từ TrainingArguments

## Sử Dụng

```bash
# Resume training
python train_ranker.py --resume_from_checkpoint experiments/Llama-2-7b-hf/ml-100k/checkpoint-x00

# Validate checkpoint
python test_checkpoint_loading.py <checkpoint_path>
```

## Lưu Ý

- Checkpoint phải chứa `adapter_model.bin` và `adapter_config.json`
- Dùng HuggingFace checkpoint directory (không phải file .pth)
- Chạy `test_checkpoint_loading.py` trước khi resume