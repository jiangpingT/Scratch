#!/bin/bash
# 合并播客音频片段

echo "🎵 正在合并播客音频..."

cd output/podcasts

# 创建文件列表
cat > filelist.txt << 'EOF'
file 'segment_000_host_a.mp3'
file 'segment_001_host_a.mp3'
file 'segment_002_host_b.mp3'
file 'segment_003_host_a.mp3'
file 'segment_004_host_b.mp3'
file 'segment_005_host_a.mp3'
EOF

# 合并音频
ffmpeg -f concat -safe 0 -i filelist.txt -c copy podcast_complete_$(date +%Y%m%d).mp3 -y

# 清理临时文件
rm filelist.txt

echo "✅ 合并完成！"
echo "📁 文件位置: output/podcasts/podcast_complete_$(date +%Y%m%d).mp3"
echo ""
echo "🎧 播放："
echo "   open output/podcasts/podcast_complete_$(date +%Y%m%d).mp3"
