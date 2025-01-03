python pth_gemm_linear.py --batch_size 10 \
    --sequence_length 936 \
    --in_channel 2880 \
    --out_channel 11520 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "ffn中的 2个升维gemm"
python pth_gemm_linear.py --batch_size 40 \
    --sequence_length 936 \
    --in_channel 2880 \
    --out_channel 2880 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "sa的q、k、v、output以及ca的q、out"

#4:ffn中的 降维的两个gemm
python pth_gemm_linear.py --batch_size 10 \
    --sequence_length 936 \
    --in_channel 11520 \
    --out_channel 2880 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "ffn中的 2个降维gemm"

#4:Temp-Attention的q、k、v以及output linear的gemm
python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 4680 \
    --in_channel 2880 \
    --out_channel 2880 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "Temp-Attention的q、k、v以及output linear的gemm"


#1:Temp-Attention的token_merge
python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 4680 \
    --in_channel 11520 \
    --out_channel 2880 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "Temp-Attention的token_merge"

#1:Temp-Attention的token_split
python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 4680 \
    --in_channel 2880 \
    --out_channel 11520 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "Temp-Attention的token_split"

#2:cross attention的 k v
python pth_gemm_linear.py --batch_size 40 \
    --sequence_length 256 \
    --in_channel 2880 \
    --out_channel 2880 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "cross attention的 k v"

#1:text prompt 4096 to 2880, once a dit model
python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 256 \
    --in_channel 4096 \
    --out_channel 2880 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "#text prompt dim 4096 to 2880, once a dit model"


python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 256 \
    --in_channel 4096 \
    --out_channel 4096 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "#text encoder gemm 1"

python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 256 \
    --in_channel 4096 \
    --out_channel 4096 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "#text encoder gemm 2"

python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 256 \
    --in_channel 4096 \
    --out_channel 10240 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "#text encoder gemm 3"

python pth_gemm_linear.py --batch_size 2 \
    --sequence_length 256 \
    --in_channel 10240 \
    --out_channel 4096 \
    --run_iter 10 \
    --warmup_iter 10 \
    --hw_tflops 165 \
    --ops gemm \
    --note "#text encoder gemm 3"