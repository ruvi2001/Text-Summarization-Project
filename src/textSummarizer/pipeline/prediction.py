from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()  #to access model and tokenizer

    def predict(self,text):
         
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        # sample_text = dataset_samsum["test"][0]["dialogue"]

        # reference = dataset_samsum["test"][0]["summary"]

        pipe = pipeline("summarization", model=self.config.model_path,tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        # print("\nReference Summary:")
        # print(reference)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        # print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])
        # print("\nModel Summary:")
        print(output)

        return output

