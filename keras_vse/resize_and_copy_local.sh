#!/usr/bin/env bash
echo "copying files to "
outdir=/export/u10/sadali/AIDA/images/GoogleImageDownload_Rus_Scenario/squared
echo $outdir
while read line
do
flink="$line"
echo $flink
imagename=$(basename "$flink")
dirname="$(dirname "$flink")"
dirname="$(basename "$dirname")"
mkdir -p  "$outdir/$dirname"
convert  -define jpeg:size=512x512  "$flink"  -thumbnail 256x256^ -gravity center -extent 256x256 "$outdir/$dirname/$imagename"
if  [[ $? -gt 0  ]]; then
echo $flink >> unconvertible_images.txt
fi
done < $1
