[
    {
        "source":"clpsych_deduped",
        "target":"multitask",
        "use_plda":true,
        "C":[5],
        "k_latent":75,
        "k_per_label":25,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"clpsych_deduped",
        "target":"wolohan",
        "use_plda":true,
        "C":[0.001],
        "k_latent":25,
        "k_per_label":50,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"clpsych_deduped",
        "target":"smhd",
        "use_plda":true,
        "C":[10],
        "k_latent":75,
        "k_per_label":50,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"multitask",
        "target":"clpsych_deduped",
        "use_plda":true,
        "C":[100],
        "k_latent":50,
        "k_per_label":75,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"multitask",
        "target":"wolohan",
        "use_plda":true,
        "C":[50],
        "k_latent":75,
        "k_per_label":75,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"multitask",
        "target":"smhd",
        "use_plda":true,
        "C":[10],
        "k_latent":200,
        "k_per_label":25,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"wolohan",
        "target":"clpsych_deduped",
        "use_plda":true,
        "C":[0.1],
        "k_latent":100,
        "k_per_label":50,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"wolohan",
        "target":"multitask",
        "use_plda":true,
        "C":[5],
        "k_latent":100,
        "k_per_label":50,        
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"wolohan",
        "target":"smhd",
        "use_plda":true,
        "C":[0.001],
        "k_latent":150,
        "k_per_label":75,        
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"smhd",
        "target":"clpsych_deduped",
        "use_plda":true,
        "C":[50],
        "k_latent":75,
        "k_per_label":100,        
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"smhd",
        "target":"multitask",
        "use_plda":true,
        "C":[10],
        "k_latent":200,
        "k_per_label":100,
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    },
    {
        "source":"smhd",
        "target":"wolohan",
        "use_plda":true,
        "C":[100],
        "k_latent":75,
        "k_per_label":25,        
        "norm":["l2"],
        "alpha":1e-2,
        "beta":1e-2
    }
]