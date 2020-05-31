#!/bin/bash

myFile="/tmp/monitor_process.time"




while ((1))
do
          if [  -f "$myFile" ]; then
          cat $myFile | while read myline
          do
            t=`date +%s`
            diff=$[ t - myline]
            echo "LINE:"$diff
            if (( $num > 300 ));
            then
                kill `ps -ef |grep process.py|awk '{print $2}'`
            fi
          done
        fi
        num=`ps -ef  | grep process |grep python | wc -l`
        echo $num
        if (( $num < 1 ));
        then
              echo "restart "
              nohup python -u /data/code/python/diandi/process.py >/tmp/process.log &
        fi
        DATE=`date "+%F %T"`
        echo $DATE
        sleep 2
done
