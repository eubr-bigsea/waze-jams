




date

runcompss  -d  --lang=python --PyObject_serialize=false \
          --appdir=/home/lucasmsp/workspace/BigSea/waze-jams/compss \
          --pythonpath=/home/lucasmsp/workspace/BigSea/waze-jams/compss \
          /home/lucasmsp/workspace/BigSea/waze-jams/compss/waze_jams.py \
          -p "/home/lucasmsp/workspace/BigSea/waze-jams/compss/prepare.m" \
          -r "/home/lucasmsp/workspace/BigSea/waze-jams/compss/runGP.m" \
          -f "/home/lucasmsp/workspace/BigSea/waze-jams/compss/sample" \
          -o "/home/lucasmsp/workspace/BigSea/waze-jams/compss/"
          -n 4
#runcompss --lang=python --log_level=info -g /home/lucasmsp/workspace/BigSea/waze-jams/compss/waze_jams.py
date
