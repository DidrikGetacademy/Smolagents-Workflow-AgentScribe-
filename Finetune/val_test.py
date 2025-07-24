
from log import validation_logger
import torch
from difflib import SequenceMatcher

def run_eval_comparison_test(trainer, sft_config, model, tokenizer, eval_dataset, num_samples=15, max_new_tokens=150, phase="Before Training"):
    model.eval()
    validation_logger(f"\n📊 Running Evaluation Test – {phase}")

    prepared_eval_dataset = trainer._prepare_dataset(
        dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
        packing=False,
        formatting_func=None,
        dataset_name=f"eval_{phase.lower().replace(' ', '_')}",
    )

    correct_count = 0
    wrong_count = 0
    total_count = min(num_samples, len(prepared_eval_dataset))

    for i in range(total_count):
        example = prepared_eval_dataset[i]
        input_ids_list = example["input_ids"]
        assistant_mask = example.get("assistant_masks", None)

        first_assistant_idx = next((idx for idx, m in enumerate(assistant_mask) if m == 1), len(input_ids_list)) if assistant_mask else len(input_ids_list)
        user_input_ids = input_ids_list[:first_assistant_idx]
        gt_assistant_ids = input_ids_list[first_assistant_idx:] if assistant_mask else []

        input_ids = torch.tensor(user_input_ids, device=model.device).unsqueeze(0)

        with torch.no_grad():
            try:
                output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
            except Exception as e:
                validation_logger(f"❌ Failed generation on example {i+1}: {e}")
                continue

        # Modellens svar
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # Ground truth svar
        expected_text = tokenizer.decode(gt_assistant_ids, skip_special_tokens=True).strip()

        # Sammenlign likhet
        similarity = SequenceMatcher(None, generated_text, expected_text).ratio()

        # 0.85+ tolkes som "riktig"
        if similarity > 0.85:
            correct_count += 1
        else:
            wrong_count += 1

        validation_logger(f"\n\n\n🔹 Example {i+1}")
        validation_logger(f"🎯 Expected: {expected_text}")
        validation_logger(f"🧠 Generated: {generated_text}")
        validation_logger(f"✅ Similarity: {similarity:.2f} -> {'✔️' if similarity > 0.85 else '❌'}")
        validation_logger("-" * 60)
        validation_logger("\n\n\n\n")

    accuracy_percent = (correct_count / total_count) * 100
    validation_logger(f"\n📈 Eval Accuracy ({phase}): {correct_count}/{total_count} = {accuracy_percent:.2f}%")
    validation_logger(f"❌ Wrong count: {wrong_count}\n")
