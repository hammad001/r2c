#!/bin/bash
if [ $1 = 'to_small' ];
then
	echo 'In to_small'
	mv train.jsonl train.jsonl.bk
	mv val.jsonl val.jsonl.bk
	mv test.jsonl test.jsonl.bk

	mv train_100.jsonl train.jsonl
	mv val_100.jsonl val.jsonl
	mv test_100.jsonl test.jsonl

elif [ $1 = 'to_big' ];
then
	echo 'In to_big'
	mv train.jsonl train_100.jsonl
	mv val.jsonl val_100.jsonl
	mv test.jsonl test_100.jsonl

	mv train.jsonl.bk train.jsonl
	mv val.jsonl.bk val.jsonl
	mv test.jsonl.bk test.jsonl

else
	echo " Wrong options provided "
fi
