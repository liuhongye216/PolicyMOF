# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /home/liuhongye/Model/MOF_llama3_1/v2-20260311-122622/checkpoint-20000 \
    --merge_lora true \
    --device_map cpu