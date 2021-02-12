#!/bin/bash
#########################################
# This script will kill all the child process id for a  given pid
#Store the current Process ID, we don't want to kill the current executing process id
CURPID=$$

_kill_process_subchildren() {

	ppid=$1
	# echo "killing subchildren of process $ppid"
	arraycounter=1

	FORLOOP=FALSE
	# Get all the child process id
	for i in `ps -ef| awk '$3 == '$ppid' { print $2 }'`
	do
	        if [ $i -ne $CURPID ] ; then
	                procid[$arraycounter]=$i
	                arraycounter=`expr $arraycounter + 1`
	                ppid=$i
	                FORLOOP=TRUE
	        fi
	done
	if [ "$FORLOOP" = "FALSE" ] ; then
	   arraycounter=`expr $arraycounter - 1`
	   ## We want to kill child process id first and then parent id's
	   while [ $arraycounter -ne 0 ]
	   do
	     kill -2 "${procid[$arraycounter]}" >/dev/tty
	     arraycounter=`expr $arraycounter - 1`
	   done
	 # exit
	fi
}
#####################################

cleanup () {
	echo "Shutting down Panda Control" >/dev/tty
	_kill_process_subchildren $PANDACTRL_PID
	_kill_process_subchildren $DRIVER_PID
	_kill_process_subchildren $REDIS_PID
	exit
}

trap cleanup SIGINT

LOCAL_PERLS2_VARS=$1
echo "Sourcing local machine config from $LOCAL_PERLS2_VARS"
source "$LOCAL_PERLS2_VARS"

DRIVER_CFG="${PERLS2_DIR}/cfg/franka-panda.yaml"
START_DRIVER_CMD="./franka_panda_driver $DRIVER_CFG"
START_REDIS_CMD="redis-server $REDIS_CONF"
START_PANDACTRL_CMD="python perls2/ctrl_interfaces/panda_ctrl_interface.py"

# Start redis-server
killall "redis-server"
sleep2
echo "Starting redis-server"
eval "${START_REDIS_CMD} &"
REDIS_PID=$!
sleep 2

# Start franka-panda-iprl driver
echo "${DRIVER_DIR}"
cd ${DRIVER_DIR}"/bin/"
eval "$START_DRIVER_CMD &"
DRIVER_PID=$! 
sleep 2

# Start Panda Ctrl Interface
cd ~
eval "$SOURCE_ENV_CMD" 
cd "${PERLS2_DIR}"
eval "$START_PANDACTRL_CMD &"
PANDACTRL_PID=$!

# idle waiting for abort from user
read -r -d '' _ </dev/tty
