SUBDIR := codex_pod_agent

.PHONY: help dev build test lint run

help:
	@echo "Use: make [dev|build|test|lint|run]"
	@echo "Forwarding to $(SUBDIR)/Makefile"

%:
	@$(MAKE) -C $(SUBDIR) $@

