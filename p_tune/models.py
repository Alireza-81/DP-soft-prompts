from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM
from src.model import GPT2Config, GPT2LMModel
import torch


def create_model(args):
    if '11b' in args.model_name:
        from ..megatron_11b.megatron_wrapper import load_megatron_lm
        print("Warning: loading MegatronLM (11B) in fp16 requires about 28G GPU memory, and may need 3-5 minutes to load.")
        return load_megatron_lm(args)
    model, _ = get_model_and_tokenizer_class(args)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        model.load_weight(torch.load(args.init_checkpoint))   
    #model = MODEL_CLASS.from_pretrained(args.model_name)
    if not args.use_lm_finetune:
        if 'megatron' in args.model_name:
            raise NotImplementedError("MegatronLM 11B is not for fine-tuning.")
        model = model.half()
    return model


def get_model_and_tokenizer_class(args):
    if 'gpt' in args.model_name:
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
        return GPT2LMModel(config), AutoTokenizer
    elif 'bert' in args.model_name:
        return AutoModelForMaskedLM, AutoTokenizer
    elif 'megatron' in args.model_name:
        return None, AutoTokenizer
    else:
        raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))


def get_embedding_layer(args, model):
    if 'roberta' in args.model_name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'bert' in args.model_name:
        embeddings = model.bert.get_input_embeddings()
    elif 'gpt' in args.model_name:
        embeddings = model.transformer.wte
    elif 'megatron' in args.model_name:
        embeddings = model.decoder.embed_tokens
    else:
        raise NotImplementedError()
    return embeddings
