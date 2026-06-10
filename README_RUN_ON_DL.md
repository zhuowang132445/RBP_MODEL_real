在 DL 服务器运行：

1. 进入项目目录
   cd /public/home/wz/workplace/cursor/modle/RBP_TRACE_V2

2. 如无环境，创建最小 venv
   python3 -m venv .venv_motif_v3
   source .venv_motif_v3/bin/activate
   pip install -U pip
   pip install -r tmp_runtime_motif_head_v3/requirements_v3.txt

3. 运行
   cd /public/home/wz/workplace/cursor/modle/RBP_TRACE_V2/tmp_runtime_motif_head_v3
   python run_motif_head_v3_no_prior_generalized.py

4. 输出目录
   /public/home/wz/workplace/cursor/modle/RBP_TRACE_V2/results/review_motif_head_v3_no_prior_generalized
