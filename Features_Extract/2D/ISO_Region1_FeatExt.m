function [Region1Features] = ISO_Region1_FeatExt(zzt)


[zzt_LAD,zzt_LCX,zzt_RCA] = Artery_Divide_Region256(zzt);

zzt_max_LAD = max(zzt_LAD);
zzt_min_LAD = min(zzt_LAD);
zzt_mean_LAD = mean(zzt_LAD);
zzt_std_LAD = std(zzt_LAD);
zzt_max_LCX = max(zzt_LCX);
zzt_min_LCX = min(zzt_LCX);
zzt_mean_LCX = mean(zzt_LCX);
zzt_std_LCX = std(zzt_LCX);
zzt_max_RCA = max(zzt_RCA);
zzt_min_RCA = min(zzt_RCA);
zzt_mean_RCA = mean(zzt_RCA);
zzt_std_RCA = std(zzt_RCA);

Region1Features = [zzt_max_LAD,zzt_min_LAD,zzt_mean_LAD,zzt_std_LAD,...
                    zzt_max_LCX,zzt_min_LCX,zzt_mean_LCX,zzt_std_LCX,...
                    zzt_max_RCA,zzt_min_RCA,zzt_mean_RCA,zzt_std_RCA];

end

function [zzt_LAD,zzt_LCX,zzt_RCA] = Artery_Divide_Region256(zzt)

zzt1_1 = zzt(129:170, 44:170);
zzt1_2 = zzt(171:256, 44:213);
zzt1_1 = zzt1_1(:);
zzt1_2 = zzt1_2(:);
zzt_LAD = vertcat(zzt1_1, zzt1_2);

zzt2 = zzt(44:128, 129:256);
zzt_LCX = zzt2(:);

zzt3 = zzt(44:128, 1:128);
zzt_RCA = zzt3(:);

end



