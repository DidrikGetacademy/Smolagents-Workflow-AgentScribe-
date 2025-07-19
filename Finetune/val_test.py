

from log import validation_logger
import torch

def print_model_output_evaltest(trainer, sft_config, model, tokenizer, eval_dataset, num_samples=3, max_new_tokens=150):

    
    model.eval()
    validation_logger("-------------------------Running an eval test after training to see model's performance..--------------------------------.\n")


    prepared_eval_dataset = trainer._prepare_dataset(
        dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
        packing=False,
        formatting_func=None,
        dataset_name="eval",
    )


    for i in range(min(num_samples, len(prepared_eval_dataset))):
        example = prepared_eval_dataset[i]
        input_ids_list = example["input_ids"]
        assistant_mask = example.get("assistant_masks", None)

  
        if assistant_mask is not None:
            first_assistant_idx = next((idx for idx, m in enumerate(assistant_mask) if m == 1), len(input_ids_list))
        else:
            first_assistant_idx = len(input_ids_list)


        user_input_ids = input_ids_list[:first_assistant_idx] 


    
        raw_input_text = tokenizer.decode(user_input_ids, skip_special_tokens=False)
        validation_logger(f"Raw input text for example {i+1}: {raw_input_text}")

    
        input_ids = torch.tensor(user_input_ids, device=model.device).unsqueeze(0)

       
        try:
            validation_logger(f"input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,             
                     )
        except Exception as e:
            validation_logger(f"🔥 model.generate failed: {e}")
            return  # Skip this example
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        validation_logger(f"DEBUG TEST OUTPUT: {output}")

        # 7) Dekode ground truth assistant (for validation_loggerging)
        if assistant_mask is not None:
            gt_assistant_ids = input_ids_list[first_assistant_idx:]
            decoded_assistant = tokenizer.decode(gt_assistant_ids, skip_special_tokens=True)
            validation_logger(f"assistant mask is not none")
        else:
            decoded_assistant = "<No assistant response in example>"
            validation_logger(f"attention mask is  none")

        # 8) Dekode modellen sin generering (kun ny generert tekst)
        gen_ids = output_ids[0][input_ids.shape[-1]:]
        model_generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 9) Print og validation_loggerge
        detailed_validation_logger = (
            f"\n🔹 Example {i+1}\n"
            f"📥 Raw model input:\n{raw_input_text}\n\n"
            f"🎯 Ground truth assistant response:\n{decoded_assistant}\n\n"
            f"🧠 Model generated response:\n{model_generated_text}\n"
            + ("-" * 60)
        )
        validation_logger(detailed_validation_logger)