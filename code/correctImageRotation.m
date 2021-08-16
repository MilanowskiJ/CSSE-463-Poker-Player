function finalImage = correctImageRotation(inputImage)

    if(numel(size(inputImage))>=3)
        grayImage = rgb2gray(inputImage);
        %grayImage = imadjust(grayImage, [0, .90]);
    else
        grayImage = inputImage;
    end

    bwTestingImage = medfilt2(grayImage);
    BW = edge(bwTestingImage, 'canny');

    [H,theta,rho] = hough(BW);
    P = houghpeaks(H,2,'threshold',ceil(0.05*max(H(:))));

    x = theta(P(:,2));
    y = rho(P(:,1));

    lines = houghlines(BW,theta,rho,P,'FillGap',5,'MinLength',25);
    figure, imshow(inputImage), hold on
    max_len = 0;
    max_th = 0;
    max_index = -1;
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
       th = lines(k).theta;
       
        plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','yellow');

       % Determine the endpoints of the longest line segment
       len = norm(lines(k).point1 - lines(k).point2);
       if ( len > max_len)
          max_len = len;
          xy_long = xy;
          max_th = th;
          max_index = k;
       end
    end
    
    if ~exist('xy_long', 'var')
        finalImage = inputImage;
        return;
    end

    second_len = 0;
    xy_second = xy_long;
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
       th = lines(k).theta;

       % Determine the endpoints of the second longest line segment
       len = norm(lines(k).point1 - lines(k).point2);
       if ((len > second_len) && (k ~= max_index))
          second_len = len;
          xy_second = xy;
       end
    end

    % highlight the longest line segment
    plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','red');
    plot(xy_second(:,1),xy_second(:,2),'LineWidth',2,'Color','red');

    centerLongX = sum(xy_long(:,1)) / 2;
    centerLongY = sum(xy_long(:,2)) / 2;

    centerShortX = sum(xy_second(:,1)) / 2;
    centerShortY = sum(xy_second(:,2)) / 2;

    centerX = (centerLongX + centerShortX)/2;
    centerY = (centerLongY + centerShortY)/2;

    plot(centerX,centerY, 'r+', 'MarkerSize',30,'LineWidth', 2, 'Color','yellow');

    finalImage = rotateAround(inputImage, centerX, centerY, max_th);
    imshow(finalImage);

end