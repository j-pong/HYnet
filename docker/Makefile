.DEFAULT_GOAL := help

# <GPU>
GPU?=0,1,2,3
DOCKER=docker
#-------------------------------------------------------------------------------
# JUPYTER=notebook
# JUPYTER=lab
#-------------------------------------------------------------------------------
ARG_IMAGE_NAME?=moneynet/cuda:base
ARG_CONTAINER_NAME?=moneynet-test
# ARG_CONTAINER_HOSTNAME?=$(shell echo $(ARG_CONTAINER_NAME) | tr a-z A-Z)
ARG_CONTAINER_HOSTNAME?=$(ARG_CONTAINER_NAME)
#-------------------------------------------------------------------------------
ARG_PUID?=$(shell id -u)
ARG_PGID?=$(shell id -g)
#-------------------------------------------------------------------------------
ARG_PORT_SSH?=32769
ARG_PORT_TENSORBOARD?=9206
#-------------------------------------------------------------------------------
ARG_WORKSPACE_HOST?=$(realpath ../..)
ARG_WORKSPACE_CONTAINER?=/home/Workspace

#*******************************************************************************
# $ make xxx
#-------------------------------------------------------------------------------
_build: ## 로컬에 도커 이미지를 만듭니다.
	docker build -t $(ARG_IMAGE_NAME) \
		-f Dockerfile .

_push: ## [관리자 전용] 이미지를 레지스트리에 push 합니다.
	docker push $(ARG_IMAGE_NAME)

pull: ## 레지스트리에서 이미지를 받아옵니다.
	docker pull $(ARG_IMAGE_NAME)

run: ## 도커 컨테이너를 실행합니다. (랜덤 포트 연결)
	$(DOCKER) run -d --restart=unless-stopped \
		--name $(ARG_CONTAINER_NAME) \
		--ipc=host \
		-h $(ARG_CONTAINER_HOSTNAME) \
		-e PUID=$(ARG_PUID) -e PGID=$(ARG_PGID) \
		-P \
		-v $(ARG_WORKSPACE_HOST):$(ARG_WORKSPACE_CONTAINER) \
		-v /etc/timezone:/etc/timezone \
		-v /etc/localtime:/etc/localtime \
		$(ARG_IMAGE_NAME)

runi: ## 도커 컨테이너를 실행합니다. (랜덤 포트 연결. 인터랙티브 모드)
	$(DOCKER) run -it --restart=unless-stopped \
		--name $(ARG_CONTAINER_NAME) \
		--ipc=host \
		-h $(ARG_CONTAINER_HOSTNAME) \
		-e PUID=$(ARG_PUID) -e PGID=$(ARG_PGID) \
		-P \
		-v $(ARG_WORKSPACE_HOST):$(ARG_WORKSPACE_CONTAINER) \
		-v /etc/timezone:/etc/timezone \
		-v /etc/localtime:/etc/localtime \
		$(ARG_IMAGE_NAME) \
		bash

runf: ## 도커 컨테이너를 실행합니다. (고정 포트 연결)
	$(DOCKER) run -d --restart=unless-stopped \
		--name $(ARG_CONTAINER_NAME) \
		--ipc=host \
		-h $(ARG_CONTAINER_HOSTNAME) \
		-e PUID=$(ARG_PUID) -e PGID=$(ARG_PGID) \
		-p $(ARG_PORT_SSH):22 \
		-p $(ARG_PORT_TENSORBOARD):6006 \
		-v $(ARG_WORKSPACE_HOST):$(ARG_WORKSPACE_CONTAINER) \
		-v /etc/timezone:/etc/timezone \
		-v /etc/localtime:/etc/localtime \
		$(ARG_IMAGE_NAME)

start: ## 컨테이너를 시작합니다.
	docker start $(ARG_CONTAINER_NAME)

stop: ## 컨테이너를 중지합니다.
	docker stop $(ARG_CONTAINER_NAME)

rm: stop ## 컨테이너를 중지하고 삭제합니다.
	docker rm $(ARG_CONTAINER_NAME)

copyauth: ## 호스트의 authrized_keys를 컨테이너에 복사 (as moneynet)
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	scp -P $(SSH_PORT) $(HOME)/.ssh/authorized_keys moneynet@$(HOST_IP):/home/moneynet/.ssh

ssh: ## 컨테이너에 SSH 연결 (as moneynet)
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	ssh moneynet@$(HOST_IP) -p $(SSH_PORT)

sshroot: ## 컨테이너에 SSH 연결 (as root)
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	ssh root@$(HOST_IP) -p $(SSH_PORT)

info: ## SSH로 컨테이너에 연결할 때 사용할 커맨드 등 출력
	$(info ----------------------------------------)
	$(info ARG_IMAGE_NAME          = $(ARG_IMAGE_NAME))
	$(info ARG_CONTAINER_NAME      = $(ARG_CONTAINER_NAME))
	$(info ARG_WORKSPACE_HOST      = $(ARG_WORKSPACE_HOST))
	$(info ARG_WORKSPACE_CONTAINER = $(ARG_WORKSPACE_CONTAINER))
	$(info ----------------------------------------)
	$(info ARG_PUID                = $(ARG_PUID))
	$(info ARG_PGID                = $(ARG_PGID))
	$(info ----------------------------------------)
	$(eval TB_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 6006/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	$(info Port(TensorBoard)=$(TB_PORT))
	$(info ----------------------------------------)
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	$(info HINT: ssh-keygen -t rsa -b 4096 -C "$$(id -un)[$$(id -u)-$$(id -g)]@$$(hostname)")
	$(info HINT: ssh-copy-id -i ~/.ssh/id_rsa.pub moneynet@$(HOST_IP) -p $(SSH_PORT))
	$(info ----------------------------------------)
	$(info ==> ssh moneynet@$(HOST_IP) -p $(SSH_PORT))
	$(info ----------------------------------------)

logs: ## 컨테이너 내부의 로그를 봅니다.
	docker logs $(ARG_CONTAINER_NAME)

env: ## EXEC: 컨테이너 내부의 도커 환경을 봅니다. (as moneynet)
	docker exec -u $(ARG_PUID):$(ARG_PGID) $(ARG_CONTAINER_NAME) env

bash: ## EXEC: 컨테이너 내부의 bash 실행 (as moneynet)
	docker exec -it -u $(ARG_PUID):$(ARG_GUID) $(ARG_CONTAINER_NAME) bash

bashroot: ## EXEC: 컨테이너 내부의 bash 실행 (as root)
	docker exec -it $(ARG_CONTAINER_NAME) bash

piplist: ## EXEC: 컨테이너 내부의 pip list 실행 (as moneynet)
	docker exec -u $(ARG_PUID):$(ARG_PGID) $(ARG_CONTAINER_NAME) pip list

cat: ## Makefile 출력
	@cat Makefile

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'
