set -e
cd ~/screen
echo "[1/3] conda env 생성..."
conda create -n fieldscreen -c conda-forge -y python=3.11 rdkit numpy scipy scikit-learn pandas matplotlib tqdm >/tmp/conda_create.log 2>&1 && echo "conda env OK" || echo "conda env FAIL (로그 /tmp/conda_create.log)"
echo "[2/3] PDBbind 업로드 완료 대기..."
while ls data/pdbbind/v2020R1/.P-L.tar.gz.* >/dev/null 2>&1; do sleep 10; done
echo "업로드 완료: $(ls -la data/pdbbind/v2020R1/P-L.tar.gz | awk "{print int(\$5/1048576)\"MB\"}")"
echo "[3/3] 압축해제..."
cd data/pdbbind/v2020R1 && tar xzf index.tar.gz && tar xzf P-L.tar.gz && echo "해제 완료" && ls -d */ | head
echo "ALL DONE"
