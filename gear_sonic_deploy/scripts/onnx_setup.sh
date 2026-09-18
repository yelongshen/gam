cd /tmp

ONNX_VERSION=1.16.3
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz"

curl -L --retry 3 --retry-delay 2 "$ONNX_URL" -o onnxruntime.tgz
tar xzf onnxruntime.tgz

sudo rm -rf /opt/onnxruntime
sudo mv onnxruntime-linux-aarch64-${ONNX_VERSION} /opt/onnxruntime

sudo ln -sf /opt/onnxruntime/lib/libonnxruntime.so /usr/local/lib/
sudo ln -sf /opt/onnxruntime/include /usr/local/include/onnxruntime
echo /opt/onnxruntime/lib | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
sudo ldconfig
