import json
import random
import os

# 定义不同任务的文件路径
TASK_FILES = {
    "generation": [
        "/home/liuhongye/material_LLM/material_LLM_experiment/data_mid_CO2/data_gene/train/gene_mix_all.jsonl"
    ],
    "classification": [
        "/home/liuhongye/material_LLM/material_LLM_experiment/data_mid_CO2/get_jsonl/train/cls_CO2_train.jsonl",
        "/home/liuhongye/material_LLM/material_LLM_experiment/data_mid_CO2/get_jsonl/test/cls_CO2_test.jsonl"
    ],
    "regression": [
        "/home/liuhongye/material_LLM/material_LLM_experiment/data_mid_CO2/get_jsonl/train/reg_reg_CO2.jsonl",
        "/home/liuhongye/material_LLM/material_LLM_experiment/data_mid_CO2/get_jsonl/test/reg_reg_CO2.jsonl"
    ]
}

OUTPUT_FILE = "/home/liuhongye/ms-swift/examples/train/multitask_gen/data.jsonl"

TASK_NAME_MAP = {
    "generation": "gen",
    "classification": "cls",
    "regression": "reg",
}


def get_label(data, task):
    """提取分类/回归任务标签，并转换为 trainer 需要的类型。"""
    for key in ("label", "labels", "score", "target", "value"):
        if key in data:
            return int(data[key]) if task == "cls" else float(data[key])
    raise ValueError(f"{task} 样本缺少 label 字段: {data}")


def normalize_to_swift_format(data, task_type):
    """
    将单条数据统一处理为 multi_task_gen 需要的格式：
    {"messages": [...], "task": "gen" | "cls" | "reg", "label": ...}
    """
    task = TASK_NAME_MAP[task_type]

    if "messages" in data:
        messages = data["messages"]
    else:
        messages = []
        system_content = data.get("system", "")
        query = data.get("query", data.get("instruction", data.get("input", "")))
        response = data.get("response", data.get("output", data.get("answer", "")))

        if system_content:
            messages.append({"role": "system", "content": system_content})
        if query:
            messages.append({"role": "user", "content": query})
        if response:
            messages.append({"role": "assistant", "content": response})

    result = {
        "messages": messages,
        "task": task,
    }

    if task in ("cls", "reg"):
        result["label"] = get_label(data, task)
        # 分类/回归样本不能把答案放在 assistant 里，否则会泄漏标签。
        result["messages"] = [msg for msg in messages if msg.get("role") != "assistant"]

    return result

def main():
    merged_data = []
    
    for task_type, files in TASK_FILES.items():
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"警告：文件不存在，已跳过 -> {file_path}")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        normalized_record = normalize_to_swift_format(record, task_type)
                        merged_data.append(normalized_record)
                        count += 1
                    except json.JSONDecodeError:
                        print(f"JSON 解析错误，跳过该行 -> {line[:50]}")
                    except (ValueError, TypeError) as e:
                        print(f"数据格式错误，跳过该行 -> {e}")
                        
            print(f"[{task_type}] 加载了 {count} 条数据来自: {os.path.basename(file_path)}")

    # 打乱数据，保证训练时模型均匀学习各个任务
    print(f"\n总计收集到 {len(merged_data)} 条数据，准备打乱合并...")
    random.shuffle(merged_data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for item in merged_data:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n多任务数据整合完毕！结果已保存至：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()