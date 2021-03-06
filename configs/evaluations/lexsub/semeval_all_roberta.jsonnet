local generator = import '../../subst_generators/lexsub/roberta.jsonnet';
local reader = import '../../dataset_readers/lexsub/semeval_all.jsonnet';

{
    class_name: "evaluations.lexsub.LexSubEvaluation",
    substitute_generator: generator,
    dataset_reader: reader,
    batch_size: 50,
    verbose: false
}
