
# XLNet
python lexsubgen/evaluations/wsi.py solve  --dataset-config-path configs/dataset_readers/wsi/semeval_2013.jsonnet --substgen-config-path configs/subst_generators/wsi/xlnet.jsonnet --clusterizer-config-path configs/clusterizers/agglo.jsonnet --run-dir debug/xlnet-semeval-2013 --force --experiment-name='wsi' --run-name='xlnet-semeval-2013' --verbose=True
python lexsubgen/evaluations/wsi.py solve  --dataset-config-path configs/dataset_readers/wsi/semeval_2010.jsonnet --substgen-config-path configs/subst_generators/wsi/xlnet.jsonnet --clusterizer-config-path configs/clusterizers/agglo.jsonnet --run-dir debug/xlnet-semeval-2010 --force --experiment-name='wsi' --run-name='xlnet-semeval-2010' --verbose=True

# XLNet+embs
python lexsubgen/evaluations/wsi.py solve  --dataset-config-path configs/dataset_readers/wsi/semeval_2013.jsonnet --substgen-config-path configs/subst_generators/wsi/xlnet_embs_se13.jsonnet --clusterizer-config-path configs/clusterizers/agglo.jsonnet --run-dir debug/xlnet-embs-semeval-2013 --force --experiment-name='wsi' --run-name='xlnet-embs-semeval-2013' --verbose=True
python lexsubgen/evaluations/wsi.py solve  --dataset-config-path configs/dataset_readers/wsi/semeval_2010.jsonnet --substgen-config-path configs/subst_generators/wsi/xlnet_embs_se10.jsonnet --clusterizer-config-path configs/clusterizers/agglo.jsonnet --run-dir debug/xlnet-embs-semeval-2010 --force --experiment-name='wsi' --run-name='xlnet-embs-semeval-2010' --verbose=True
