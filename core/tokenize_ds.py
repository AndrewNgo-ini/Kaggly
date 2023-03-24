%%writefile tokenize_ds.py

from transformers import AutoTokenizer
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

from cfg import CFG


def main(show_progress_bars=True):
    
    if not show_progress_bars:
        # Progress bars can be distracting
        disable_progress_bar()
    
    def tokenize(batch, tokenizer, topic_cols, content_cols, max_length):
        """
        Tokenizes the dataset on the specific columns, truncated/padded to a max length.
        Adds the suffix "_content" to the input ids and attention mask of the content texts.

        Returns dummy labels that make the evaluation work in Trainer.
        """
        sep = tokenizer.sep_token

        topic_texts = [sep.join(cols) for cols in zip(*[batch[c] for c in topic_cols])]
        content_texts = [sep.join(cols) for cols in zip(*[batch[c] for c in content_cols])]

        tokenized_topic = tokenizer(
            topic_texts, truncation=True, max_length=max_length, padding=False
        )

        tokenized_content = tokenizer(
            content_texts, truncation=True, max_length=max_length, padding=False
        )

        # Remove token_type_ids. They will just cause errors.
        if "token_type_ids" in tokenized_topic:
            del tokenized_topic["token_type_ids"]
            del tokenized_content["token_type_ids"]

        return {
            **{f"{k}_a": v for k, v in tokenized_topic.items()},
            **{f"{k}_b": v for k, v in tokenized_content.items()},
            "labels": [1] * len(tokenized_topic.input_ids), # placeholder for Trainer
        }


    def get_tokenized_ds(ds, tokenizer, max_length=64, debug=False):

        if debug:
            ds = ds.shuffle().select(range(5000))

        tokenized_ds = ds.map(
            tokenize,
            batched=True,
            fn_kwargs=dict(
                tokenizer=tokenizer,
                topic_cols=[f"topic_{c}" for c in CFG.topic_cols],
                content_cols=[f"content_{c}" for c in CFG.content_cols],
                max_length=max_length,
            ),
            remove_columns=ds.column_names,
            num_proc=CFG.num_proc,
        )

        return tokenized_ds

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    combined = load_dataset("parquet", data_files="combined2.pq", split="train")
    tokenized_ds = get_tokenized_ds(combined, tokenizer, max_length=CFG.max_length, debug=CFG.debug)
    tokenized_ds.to_parquet(CFG.tokenized_ds_name)



if __name__ == "__main__":

    import fire
    # fire makes it really easy to pass args
    fire.Fire(main)