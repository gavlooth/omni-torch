CXX = g++
CXXFLAGS = -O2 -fPIC -std=c++17
TORCH_DIR = /opt/libtorch

TORCH_CFLAGS = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
TORCH_LDFLAGS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_DIR)/lib

BUILD_DIR = build
OMNI_BIN ?= omni
OMNI_LD = LD_LIBRARY_PATH=$(BUILD_DIR):$(TORCH_DIR)/lib:/usr/local/lib

.PHONY: all clean test test-hardening test-hardening-host test-hardening-container test-all conformance executor-test xor diffusion diffusion-llm

all: $(BUILD_DIR)/libtorch_omni.so

$(BUILD_DIR)/libtorch_omni.so: csrc/torch_shim.cpp csrc/torch_shim.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared $(TORCH_CFLAGS) -o $@ $< $(TORCH_LDFLAGS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: $(BUILD_DIR)/libtorch_omni.so
	$(OMNI_LD) $(OMNI_BIN) src/main.omni
	$(OMNI_LD) $(OMNI_BIN) test_libtorch_conformance.omni
	$(OMNI_LD) $(OMNI_BIN) test_executor_skeleton.omni

test-hardening-host: $(BUILD_DIR)/libtorch_omni.so
	$(OMNI_LD) $(OMNI_BIN) test_m6_hardening.omni

test-hardening:
	@if [ "$(OMNI_TORCH_ALLOW_HOST_HARDENING)" = "1" ]; then \
		$(MAKE) test-hardening-host; \
	else \
		echo "test-hardening is container-only by default (host run disabled)."; \
		echo "Run: make test-hardening-container"; \
		echo "Override (unsafe): OMNI_TORCH_ALLOW_HOST_HARDENING=1 make test-hardening"; \
		exit 1; \
	fi

test-hardening-container:
	./scripts/run_hardening_container.sh

test-all: test test-hardening

executor-test: $(BUILD_DIR)/libtorch_omni.so
	$(OMNI_LD) $(OMNI_BIN) test_executor_skeleton.omni

conformance: $(BUILD_DIR)/libtorch_omni.so
	$(OMNI_LD) $(OMNI_BIN) test_libtorch_conformance.omni

xor: $(BUILD_DIR)/libtorch_omni.so
	$(OMNI_LD) $(OMNI_BIN) examples/xor_nn.omni

diffusion: $(BUILD_DIR)/libtorch_omni.so
	$(OMNI_LD) $(OMNI_BIN) examples/diffusion_2d.omni

diffusion-llm: $(BUILD_DIR)/libtorch_omni.so
	$(OMNI_LD) $(OMNI_BIN) examples/diffusion_llm.omni

clean:
	rm -rf $(BUILD_DIR)
