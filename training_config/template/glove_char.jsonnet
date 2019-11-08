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
    "tokenizer": {
      [std.extVar("task")]: std.extVar("tokenizer")
    },
    "sentencizer": {
      [std.extVar("task")]: std.extVar("sentencizer")
    },
    "eval_span_pair_skip": {
      "dp": ["punct"]
    },
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 5
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
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
          "trainable": false
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 16
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 50,
            "ngram_filter_sizes": [3, 4, 5],
            "conv_layer_activation": "relu"
          }
        }
      }
    },
    "span_pair_prediction_method": "mlp",
    "context_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 450,
      "hidden_size": 256,
      "num_layers": 3,
      "dropout": 0.3
    },
    "span_layer": {
      [std.extVar("task")]: {
        "input_dim": if std.extVar("use_attentive_span_repr") then 450 + 512 * 2 + 64 else 512 * 2 + 64,
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
          "input_dim": if std.extVar("use_attentive_span_repr") then 450 + 512 * 2 + 64 else 512 * 2 + 64,
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
    "max_span_width": 30,
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
    "batch_size": std.extVar("batch_size"),
    "batch_size_per_task": {
      [std.extVar("task")]: std.extVar("batch_size")
    },
    "max_total_seq_len": {
      [std.extVar("task")]: std.extVar("max_total_seq_len")
    },
    "biggest_batch_first": true,
    //"max_instances_in_memory": 20000
  },
  "trainer": {
    "num_epochs": 200,
    "grad_norm": 5.0,
    "patience" : 10,
    "num_serialized_models_to_keep": 10,
    "cuda_device" : 0,
    "validation_metric": "+" + std.extVar("validation_metric"),
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 3
    },
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  },
  "evaluate_on_test": true,
  "vocabulary": {
    "directory_path": "output/all_vocab/vocabulary/"
  }
}
