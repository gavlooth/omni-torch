CXX = g++
CXXFLAGS = -O2 -fPIC -std=c++17
TORCH_DIR = /opt/libtorch

TORCH_CFLAGS = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
TORCH_LDFLAGS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_DIR)/lib

BUILD_DIR = build

.PHONY: all clean test xor

all: $(BUILD_DIR)/libtorch_omni.so

$(BUILD_DIR)/libtorch_omni.so: csrc/torch_shim.cpp csrc/torch_shim.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared $(TORCH_CFLAGS) -o $@ $< $(TORCH_LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: $(BUILD_DIR)/libtorch_omni.so
	LD_LIBRARY_PATH=$(TORCH_DIR)/lib:$(BUILD_DIR) omni src/main.omni

xor: $(BUILD_DIR)/libtorch_omni.so
	LD_LIBRARY_PATH=$(TORCH_DIR)/lib:$(BUILD_DIR) omni examples/xor_nn.omni

clean:
	rm -rf $(BUILD_DIR)
