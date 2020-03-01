function alpha_blend_news(video, starti, framenum)
%video = 'Learn_English';
%starti = 357; % choose 400, 300 for training render-to-video, 100 for testing
%framenum = 400;

srcdir = ['render/',video,'/'];
srcdir2 = ['render/',video,'/'];
tardir = ['render/',video,'/bm/'];
files = dir(fullfile(srcdir,'*.png'));
t1=tic;
if ~exist(tardir)
    mkdir(tardir);
end
for i = starti:(starti+framenum-1)
    file = ['frame',num2str(i),'.png'];
    im1 = imread(fullfile(srcdir,file));
    [im2,~,trans] = imread(fullfile(srcdir2,[file(1:end-4),'_rendernew.png']));
    [B,L] = bwboundaries(trans);
    %imshow(label2rgb(L,@jet,[.5,.5,.5]));
    trans(L>=2) = 255;
    %figure;imshow(trans);
    trans = double(trans)/255;
    im3 = double(im1).*(1-trans) + double(im2).*trans;
    im3 = uint8(im3);
    imwrite(im3,fullfile(tardir,[file(1:end-4),'_render_bm.png']));
end
toc(t1)%1094.343765 seconds