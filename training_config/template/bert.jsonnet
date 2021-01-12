local bert = std.extVar("bert");
local pretrained_model_vocab = if bert == "bert-base-cased" then "bert-base-cased"
  else if bert == "bert-base-uncased" then "bert-base-uncased"
  else if bert == "bert-large-uncased" then "bert-large-uncased"
  else if bert == "span-bert-base" then "bert-base-cased"
  else if bert == "span-bert-large" then "bert-large-cased"
  else "bert-base-uncased";
local do_lowercase = if bert == "bert-base-cased" then false
  else if bert == "bert-base-uncased" then true
  else if bert == "bert-large-uncased" then true
  else if bert == "span-bert-base" then false
  else if bert == "span-bert-large" then false
  else true;
local pretrained_model = if bert == "bert-base-cased" then "bert-base-cased"
  else if bert == "bert-base-uncased" then "bert-base-uncased"
  else if bert == "bert-large-uncased" then "bert-large-uncased"
  else if bert == "span-bert-base" then "pretrain/spanbert/spanbert_hf_base/"
  else if bert == "span-bert-large" then "pretrain/spanbert/spanbert_hf/"
  else "bert-base-uncased";
local bert_dim = if bert == "bert-base-cased" then 768
  else if bert == "bert-base-uncased" then 768
  else if bert == "bert-large-uncased" then 1024
  else if bert == "span-bert-base" then 768
  else if bert == "span-bert-large" then 1024
  else 768;
local cuda = if bert == "bert-base-cased" then 0
  else if bert == "bert-base-uncased" then 0
  else if bert == "bert-large-uncased" then [0,1]
  else if bert == "span-bert-base" then 0
  else if bert == "span-bert-large" then [0,1]
  else 0;
local num_gpu = if bert == "bert-base-cased" then 1
  else if bert == "bert-base-uncased" then 1
  else if bert == "bert-large-uncased" then 2
  else if bert == "span-bert-base" then 1
  else if bert == "span-bert-large" then 2
  else 0;

local head_div = if std.extVar("use_head_attentive_span_repr") then 4 else 1;
local span_dim1 = if std.extVar("use_attentive_span_repr") then bert_dim / head_div else 0;
local span_dim2 = if std.extVar("use_context_layer") then 512 * 2 + 64 else 0;
local span_dim = span_dim1 + span_dim2;

{
  "dataset_reader": {
    "type": "brat",
    "default_task": "default_task",
    "task_sample_rate": [1],
    "restart_file": false,
    "use_neg": {
      [std.extVar("task")]: std.extVar("use_neg")
    },
    "max_span_width": {
      [std.extVar("task")]: std.extVar("max_span_width")
    },
    "max_sent_len": {
      [std.extVar("task")]: std.extVar("max_sent_len")
    },
    "max_num_sample": {
      [std.extVar("task")]: std.extVar("max_num_sample")
    },
    "tokenizer": {
      [std.extVar("task")]: std.extVar("tokenizer")
    },
    "sentencizer": {
      [std.extVar("task")]: std.extVar("sentencizer")
    },
    "eval_span_pair_skip": {
      "dp": ["punct"],
      "dp_conll": ["punct"]
    },
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": pretrained_model_vocab,
        "use_starting_offsets": true,
        "do_lowercase": do_lowercase,
        "max_pieces": 512,
        "truncate_long_sequences": false
      }
    },
    "lazy": false
  },
  "train_data_path": std.extVar("task") + "|true|data/" + std.extVar("data_dir") + "/train/",
  "validation_data_path": std.extVar("task") + "|true|data/" + std.extVar("data_dir") + "/dev/",
  "test_data_path": std.extVar("task") + "|true|data/" + std.extVar("data_dir") + "/test/",
  "model": {
    "type": "brat",
    "task_list": [std.extVar("task")],
    "task_loss": {
      [std.extVar("task")]: std.split(std.extVar("task_loss"), "-")
    },
    "task_loss_reduction": {
      [std.extVar("task")]: "sum"
    },
    "truncate_span_loss": {
      [std.extVar("task")]: std.extVar("truncate_span_loss")
    },
    "spans_per_word": {
      [std.extVar("task")]: std.extVar("spans_per_word")
    },
    "pair_ind_method": {
      [std.extVar("task")]: if std.extVar("pair_ind_method") != "null" then std.extVar("pair_ind_method")
    },
    "special_loss": {
      [std.extVar("task")]: std.extVar("special_loss")
    },
    "special_metric": {
      "wlp": [],
      "ner": [],
      "srl": [],
      "dp": [],
      "oie": [],
      "coref": ["coref", "mr"],
      "rc": ["semeval_2010"],
      "semeval14_st2": [],
      "consti": ["bracket"],
      "pos": [],
      "dp_conll": [],
      "consti_conll": ["bracket"],
      "pos_conll": [],
      "orl": ["binary_sp_prf"]
    },
    "use_attentive_span_repr": std.extVar("use_attentive_span_repr"),
    "use_head_attentive_span_repr": std.extVar("use_head_attentive_span_repr"),
    "attentive_after_context": std.extVar("attentive_after_context"),
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
      },
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained-multitask",
          "pretrained_model": pretrained_model,
          "requires_grad": true,
          "top_layer_only": true,
          "use_middle_layer": std.extVar("use_middle_layer")
        }
      }
    },
    "num_order": std.extVar("num_order"),
    "span_pair_prediction_method": "mlp",
    "use_context_layer": std.extVar("use_context_layer"),
    "context_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": bert_dim,
      "hidden_size": 256,
      "num_layers": 3,
      "dropout": 0.3
    },
    "span_layer": {
      [std.extVar("task")]: {
        "input_dim": span_dim,
        "num_layers": 2,
        "hidden_dims": 128,
        "activations": "relu",
        "dropout": 0.3
      }
    },
    "span_pair_layer": {
      [std.extVar("task")]: {
        "separate": true,
        "dim_reduce_layer": {
          "input_dim": span_dim,
          "num_layers": 1,
          "hidden_dims": 512,
          "activations": "relu",
          "dropout": 0.3
        },
        "combine": "coref",
        "dist_emb_size": 64,
        "repr_layer": {
          "input_dim": 512 * 3 + 64,
          "num_layers": 2,
          "hidden_dims": 128,
          "activations": "relu",
          "dropout": 0.3
        }
      }
    },
    "span_width_embedding_dim": 64,
    "max_span_width": std.extVar("max_span_width"),
    "bucket_widths": std.extVar("bucket_widths"),
    "lexical_dropout": 0.5,
    "regularizer": [
      ["scalar_parameters", {"type": "l2", "alpha": 0.001}]
    ]
  },
  "iterator": {
    "type": "ada_balanced_bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "task_namespace": "task_labels",
    "num_interleave_per_task": {
      [std.extVar("task")]: std.extVar("batch_size")
    },
    "batch_size": std.extVar("batch_size") / num_gpu,
    "batch_size_per_task": {
      [std.extVar("task")]: std.extVar("batch_size") / num_gpu
    },
    "max_total_seq_len": {
      [std.extVar("task")]: std.extVar("max_total_seq_len") / num_gpu
    },
    "biggest_batch_first": true
  },
  "trainer": {
    "num_epochs": std.extVar("num_epochs"),
    "grad_norm": 5.0,
    "patience" : std.extVar("patience"),
    "num_serialized_models_to_keep": 1,
    "cuda_device" : cuda,
    "validation_metric": "+" + std.extVar("validation_metric"),
    "optimizer": {
      "type": "bert_adam",
      "lr": std.extVar("lr"),
      "warmup": std.extVar("warmup"),
      "t_total": std.extVar("step_per_epoch") * std.extVar("num_epochs"),
      "schedule": "warmup_linear"
    }
  },
  "evaluate_on_test": true,
  "vocabulary": {
    "directory_path": if std.extVar("vocab") != "null" then std.extVar("vocab")
  }
}
