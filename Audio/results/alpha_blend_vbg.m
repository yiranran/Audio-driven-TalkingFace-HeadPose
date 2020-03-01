function alpha_blend_vbg(srcdirbg,srcdir)
	disp(srcdirbg);
	files = dir(fullfile(srcdir,'*.png'));
	for i = 1:length(files)
	    file = files(i).name;
		if length(file) >= 10 && strcmp(file(end-9:end),'_blend.png')
			continue;
		end
		if length(file) >= 11 && strcmp(file(end-10:end),'_blend2.png')
			continue;
		end
		%if exist(fullfile(srcdir,[file(1:end-4),'_blend2.png']))
		%	continue;
		%end
		im1 = imread(fullfile(srcdirbg,file));
		%disp(fullfile(srcdirbg,file));
	    [im2,~,trans] = imread(fullfile(srcdir,file));
	    [~,L] = bwboundaries(trans);
	    trans(L>=2) = 255;

	    trans = double(trans)/255;
	    im3 = double(im1).*(1-trans) + double(im2).*trans;
	    im3 = uint8(im3);
	    tarname = [file(1:end-4),'_blend2.png'];
	    %disp(tarname);
	    imwrite(im3,fullfile(srcdir,tarname));
	end
end