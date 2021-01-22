local reader = import '../dataset_readers/lexsub/semeval_all.jsonnet';
local post_processing = import '../subst_generators/post_processors/lower_nltk_spacy.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: [
        {
            class_name: "pre_processors.base_preprocessors.AndPreprocessor"
        }
    ],
    prob_estimator: {
        class_name: "prob_estimators.xlmr_estimator.XLMRProbEstimator",
        model_name: "xlm-roberta-large",
        coef_2_mask: {
            class_name: "LogspaceHyperparam",
            start: -1,
            end: 1,
            size: 7,
            base: 100,
            name: "coef_2"
        },
        coef_3_mask: {
            class_name: "LogspaceHyperparam",
            start: -1,
            end: 1,
            size: 5,
            base: 100,
            name: "coef_3"
        },
        coef_4_mask: {
            class_name: "LogspaceHyperparam",
            start: -1,
            end: 1,
            size: 5,
            base: 100,
            name: "coef_4"
        },
        coef_5_mask: {
            class_name: "LogspaceHyperparam",
            start: -1,
            end: 1,
            size: 5,
            base: 100,
            name: "coef_5"
        },
        cuda_device: 0
    },
    post_processing: post_processing,
    top_k: 10,
}
