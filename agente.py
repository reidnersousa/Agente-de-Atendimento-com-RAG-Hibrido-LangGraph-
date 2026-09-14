
import gc
import torch
import kagglehub
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3:
    def __init__(
        self,
        model_handle: str = "qwen-lm/qwen-3/transformers/1.7b",
        clear_gpu: bool = True,
    ):
        """
        Load Qwen3 model from KaggleHub.

        Parameters
        ----------
        model_handle : str
            KaggleHub model identifier.
        clear_gpu : bool
            Whether to clear GPU cache before loading.
        """

        if clear_gpu:
            self.clear_gpu()

        print("Downloading/loading model from KaggleHub...")
        self.model_path = kagglehub.model_download(model_handle)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model.eval()

        print("Model loaded successfully!")

    @staticmethod
    def clear_gpu():
        """
        Attempt to free GPU memory held by this process.
        """
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("GPU cache cleared.")

    def generate(self,prompt: str,max_new_tokens: int = 50,
        #max_new_tokens: int = 32768,
        #enable_thinking: bool = True,
        enable_thinking: bool = False,
    ):
        """
        Generate a response and separate thinking content.
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )

        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()

        # Parse thinking section
        try:
            index = (
                len(output_ids)
                - output_ids[::-1].index(151668)
            )
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index],
            skip_special_tokens=True,
        ).strip("\n")

        content = self.tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True,
        ).strip("\n")

        return {
            "thinking": thinking_content,
            "response": content,
        }

    def unload(self):
        """
        Explicitly unload model from GPU.
        """
        del self.model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("Model unloaded.")

llm = Qwen3()

