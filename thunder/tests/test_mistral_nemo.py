import thunder
import thunder.tests.framework
import torch
import transformers
import datasets


@thunder.tests.framework.requiresCUDA
def test_thunderfx_mistral_nemo_small():
    """
    Runs a small version of Mistral-NeMo

    This is largely based on code from Alexandros Koumparoulis.
    """
    model_id = "mistralai/Mistral-Nemo-Base-2407"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=False,
    )

    # Setup a "small" version of NeMo-Mistral that does not require downloading
    # weights. This is not a configuration that is worth benchmarking.
    # This was created by using
    #   MistralConfig(num_hidden_layers=2, max_position_embeddings=1024)
    # and then manually diffing that returned object with:
    #   transformers.AutoConfig.from_pretrained(model_id)
    # until they lined up.
    config = transformers.models.mistral.configuration_mistral.MistralConfig(
        num_hidden_layers=2,
        torch_dtype=torch.bfloat16,
        max_position_embeddings=1024,
        architectures=["MistralForCausalLM"],
        hidden_size=5120,
        rms_norm_eps=1e-05,
        rope_theta=1000000.0,
        sliding_window=None,
        vocab_size=131072,
        head_dim=128,
        _name_or_path=model_id,
    )
    model = transformers.AutoModelForCausalLM.from_config(config)
    device = torch.device("cuda")
    model.to(device)
    mdl = torch.compile(model, backend=thunder.dynamo.ThunderCompiler())
    del model

    # Add a padding token to the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        mdl.resize_token_embeddings(len(tokenizer))

    dataset = datasets.load_dataset("tiny_shakespeare", split="train",
                                    trust_remote_code=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Convert the dataset to PyTorch format and specify columns to return as tensors
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1, shuffle=True)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(mdl.parameters(), lr=5e-5)
    num_epochs = 3
    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader),
    )

    mdl.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            # Move input tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = mdl(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            print(loss)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate
            lr_scheduler.step()
