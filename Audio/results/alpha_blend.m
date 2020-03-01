function alpha_blend(person,srcdir)
	im1 = imread(fullfile(pwd,'../../Deep3DFaceReconstruction/output/render',person));
	files = dir(fullfile(srcdir,'*.png'));
	for i = 1:length(files)
	    file = files(i).name;
		if length(file) >= 10 && strcmp(file(end-9:end),'_blend.png')
			continue;
		end
		if exist(fullfile(srcdir,[file(1:end-4),'_blend.png']))
			continue;
		end
	    [im2,~,trans] = imread(fullfile(srcdir,file));
	    [~,L] = bwboundaries(trans);
	    trans(L>=2) = 255;

	    trans = double(trans)/255;
	    im3 = double(im1).*(1-trans) + double(im2).*trans;
	    im3 = uint8(im3);
	    disp([file(1:end-4),'_blend.png']);
	    imwrite(im3,fullfile(srcdir,[file(1:end-4),'_blend.png']));
	end
end