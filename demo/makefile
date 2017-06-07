SHELL=/bin/bash
PY=python3 -B

.PHONY: demo

help:
	@echo "---------------------------------------------"
	@echo "  Available make targets"
	@echo "---------------------------------------------"
	@echo
	@echo "  demo                       Run the demo!"
	@echo
	@echo
	@echo "Full dataset experiments"
	@echo "  full_all_info              Run configured embeddings on full dataset with All-Info setting"
	@echo "  full_multi_answer          Run configured embeddings on full dataset with Multi-Answer setting"
	@echo "  full_single_answer         Run configured embeddings on full dataset with Single-Answer setting"
	@echo
	@echo
	@echo "Unigram subset experiments"
	@echo "  unigrams_all_info          Run configured embeddings on unigram dataset with All-Info setting, using unigram candidates"
	@echo "  unigrams_multi_answer      Run configured embeddings on unigram dataset with Multi-Answer setting, using unigram candidates"
	@echo "  unigrams_single_answer     Run configured embeddings on unigram dataset with Single-Answer setting, using unigram candidates"
	@echo 
	@echo "  unigram_mwe_all_info       Run configured embeddings on unigram dataset with All-Info setting, using MWE candidates"
	@echo "  unigram_mwe_multi_answer   Run configured embeddings on unigram dataset with Multi-Answer setting, using MWE candidates"
	@echo "  unigram_mwe_single_answer  Run configured embeddings on unigram dataset with Single-Answer setting, using MWE candidates"
	@echo


demo: full_all_info full_multi_answer full_single_answer

### Full dataset ######################################

full_all_info:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=All-Info \
		--predictions-file=$${DATA}/logs/full.all_info.predictions.log \
		-l $${DATA}/logs/full.all_info.log \
		$${DATA}/BMASS/BMASS_all_info.txt \
		$${DATA}/results/all_info

full_multi_answer:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=Multi-Answer \
		--predictions-file=$${DATA}/logs/full.multi_answer.predictions.log \
		-l $${DATA}/logs/full.multi_answer.log \
		$${DATA}/BMASS/BMASS_multi_answer.txt \
		$${DATA}/results/multi_answer

full_single_answer:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=Single-Answer \
		--predictions-file=$${DATA}/logs/full.single_answer.predictions.log \
		-l $${DATA}/logs/full.single_answer.log \
		$${DATA}/BMASS/BMASS_single_answer.txt \
		$${DATA}/results/single_answer





### Unigram data only #################################

unigrams_all_info:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=All-Info \
		--unigrams \
		--predictions-file=$${DATA}/logs/unigrams.all_info.predictions.log \
		-l $${DATA}/logs/unigrams.all_info.log \
		$${DATA}/BMASS/Unigram_Data/BMASS_unigrams_all_info.txt \
		$${DATA}/results/unigrams/all_info

unigrams_multi_answer:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=Multi-Answer \
		--unigrams \
		--predictions-file=$${DATA}/logs/unigrams.multi_answer.predictions.log \
		-l $${DATA}/logs/unigrams.multi_answer.log \
		$${DATA}/BMASS/Unigram_Data/BMASS_unigrams_multi_answer.txt \
		$${DATA}/results/unigrams/multi_answer

unigrams_single_answer:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=Single-Answer \
		--unigrams \
		--predictions-file=$${DATA}/logs/unigrams.single_answer.predictions.log \
		-l $${DATA}/logs/unigrams.single_answer.log \
		$${DATA}/BMASS/Unigram_Data/BMASS_unigrams_single_answer.txt \
		$${DATA}/results/unigrams/single_answer




### Unigram data w/ MWE candidates ####################

unigram_mwe_all_info:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=All-Info \
		--unigram-mwe-compare \
		--predictions-file=$${DATA}/logs/unigram_mwe.all_info.predictions.log \
		-l $${DATA}/logs/unigram_mwe.all_info.log \
		$${DATA}/BMASS/Unigram_Data/BMASS_unigrams_all_info.txt \
		$${DATA}/results/unigram_mwe/all_info

unigram_mwe_multi_answer:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=Multi-Answer \
		--unigram-mwe-compare \
		--predictions-file=$${DATA}/logs/unigram_mwe.multi_answer.predictions.log \
		-l $${DATA}/logs/unigram_mwe.multi_answer.log \
		$${DATA}/BMASS/Unigram_Data/BMASS_unigrams_multi_answer.txt \
		$${DATA}/results/unigram_mwe/multi_answer

unigram_mwe_single_answer:
	@set -e; \
	DATA=$$(${PY} -m config DATA); \
	${PY} -m analogy_task.experiments_for_paper \
		--setting=Single-Answer \
		--unigram-mwe-compare \
		--predictions-file=$${DATA}/logs/unigram_mwe.single_answer.predictions.log \
		-l $${DATA}/logs/unigram_mwe.single_answer.log \
		$${DATA}/BMASS/Unigram_Data/BMASS_unigrams_single_answer.txt \
		$${DATA}/results/unigram_mwe/single_answer
