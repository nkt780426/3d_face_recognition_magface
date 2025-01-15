
# Radiometric Concepts | Radiometry and Reflectance

## Mục tiêu

Mục tiêu: Học các khái niệm được định nghĩa liên quan đến radiometric (đo bức xạ, đo cường độ ánh sáng), hiểu được các khái niệm này phẩn nào đó sẽ giúp ta tái tạo được source of light từ Image Intensity giúp mô hình 3D song động hơn. 

Nói đơn giản ta sẽ được học về các đại lượng như
- công suất của nguồn (Radiant Intensity)
- lượng ánh sáng mà bề mặt nhận được từ nguồn/đơn vị diện tích: Đại lượng đặc trưng, chỉ cần nhân diện tích bề mặt là tính được tổng lượng ánh sáng bề mặt nhận được.
- lượng ánh sáng mà bề mặt sẽ phát ra theo 1 hướng bất kỳ: theo quy luật phân bổ nào đó, chỉ cần biết hướng quan sát, pháp tuyến bề mặt của điểm và góc tạo bởi điểm đó với mắt người/camera.
=> Bài này nói từ lượng ánh sáng (bức xạ) từ nguồn cho đến tổng lượng ánh sáng mà điểm trong scene phát ra mà mắt/camera thu được

(Bài sau tìm hiểu tiếp về cách ánh sáng mà người quan sát/camera thu được từ công thức này)

## Các khái niệm

1. Angle (2D)

    ![Angle 2D](./images/2.%20Angle2D.png)

	Góc dphi được tính bang góc chắn cung dl/bán kính đường tròn 
	Đơn vị góc là radian (dl là m, r cũng là m do đó phải đặt ra 1 đơn vị mới đại diện cho những gì tính được)

2. Soild Angle (3D): góc đặc chắn bởi hình cầu.

    ![Angle 3D](./images/2.%20SoildAngle3D.png)

    Đang có 1 điểm, giả sử điểm này đang nhìn 1 diện tích vô cùng nhỏ dA nào đó 1 góc dw
	Từ diện tích dA ta tính được diện tích rút gọn dA' (foreshortened area): là cái hình tam giác có đỉnh là điểm đó và đáy là đường kính diện tích dA.
	Do góc vô cùng nhỏ nên công thức trên vẫn đúng, tuy nhiên nếu dw lớn thì góc này sẽ được chắn bởi hình cầu chứ không phải là hình tròn => Bản chat góc này được định nghĩa là nhìn hình cầu nhưng do dA quá nhỏ nên ta có thể coi nò là hình tròn. Lúc này ta phải dung tích phân tính điện tích hình cầu bị chắn bởi góc thay cho dA'

3. Light Flux: Thông lượng ánh sáng của nguồn

    ![Flux](./images/2.%20Light%20Flux.png)

	Là lượng ánh sáng phát ra từ nguồn dưới 1 góc soild angle dw
	Đơn vị là watts

4. Radiant Intensity (Cường độ sáng của nguồn)

    ![Radiant Intensity](images/2.%20Radient%20Intensity.png)

    Chính là lượng ánh sáng mà nguồn phát ra / 1 đơn vị góc (soild angle = 1 đơn vị là pi, soild angle = 2pi là 360 độ).
	Ký hiệu đại lượng này là J và là hang số với mỗi nguồn sáng trong quá trình chụp ảnh.

5. Surface Irraiance (cường độ chiếu sáng bề mặt của nguồn hay độ sáng của bề mặt)

    ![Surface Irradiance](images/2.%20Surface%20Irradiance.png)
	
    Sau khi đã xác định được radiant Intensity (độ sáng của nguồn), ta tò mò muốn biết khả năng chiếu sáng của nguồn lên bề mặt là bao nhiêu khi nó cách bề mặt 1 khoảng r. (hiểu nôm na là khả năng phát sáng bề của bề mặt nhờ nguồn sáng. Ta nhìn thấy vật khi có ánh sáng từ vật chiếu vào mắt ta)

	Surface Irraiance (E) là thông lượng ánh sáng chiếu vào bề mặt hay có bao nhiêu ánh sáng được chiếu vào bề mặt.
	Giả sử trong môi trường lý tưởng, năng lượng được bảo tồn thì tất cả thông lượng ánh sáng sẽ được đập vào bề mặt dA nên ta chỉ cần lấy lượng ánh sáng chia cho diện tích sẽ ra.
	Ta biến đổi chút sẽ thu được kết quả cuối cùng. Từ kết quả này ta kết luận:
		Độ sáng của bề mặt E phụ thuộc vào phi và r do J là hang số
		E lớn nhất khi nguồn sáng đặt vuông góc với bề mặt (phi trùng với vector pháp tuyến tại điểm xét) và càng nhỏ theo bình phương khoảng cách.

6. Surface Radiance (độ phát xạ bề mặt)

    ![Surface Radiance](images/2.%20Sufface%20Radiance.png)

    Surface Irraiance là cường độ phát sáng của bề mặt. Tuy nhiên bản thân mắt ta/camera lại được đặt cách vật 1 khoảng cách xa và hướng cũng khác và có hình dạng là tròn - Tức mắt ta có thể nhận nhiều tia tán xạ từ bề mặt. 
    => Ta cần tìm mối liên hệ/công thức giữa lượng ánh sáng đập vào mắt ta và Surface Irraiance (khả năng phát sáng của bề mắt)
	
    Surface Radiance (ký hiệu là L) là lượng ánh sáng mà 1 bề mặt phát ra theo 1 hướng nhất định trong 1 đơn vị góc nhất định (1 rad) nói nôm na là lượng ánh sáng mà mắt ta nhận được khi đứng cách bề mặt phát sáng.

	Hiển nhiên nếu ta để cảm biến máy ảnh càng xa vật thể thì ánh sáng mà bề mặt phát ra mà cảm biến thu được sẽ càng nhỏ do góc dw nhỏ đi => Thông lượng ánh sáng đập vào ít đị.
	
	Tuy nhiên đây chỉ là ta xét surface là 1 điểm. Thực tế surface sẽ lớn hơn nhiều hay tổng hợp nhiều điểm. Do đó lượng ánh sáng mà cảm biến thu được sẽ bị cộng dồn và trở nên nhiều hơn. Vì vậy ta cần 1 cách nào đó để chuẩn hóa L theo cả diện tích bề mặt cũng như soild angle để dễ dàng tính toán khi diện tích bề mặt và góc soild thay đổi.
	Có thể thấy L phụ thuộc vào 
		góc theta, ánh sáng ược tán xạ vào các góc theta gần 0 và gần như không tán xạ khi theta gần 90 độ
		đặc tính phản xạ của surface: lượng ánh sáng mà bề mặt nhận được luôn bị bề mặt vật thể hấp thụ 1 phần và tán xạ phần còn lại. 
        Tùy thuộc vào bề mặt vật liệu, mỗi vật liệu sẽ có 1 hệ số phản xạ khác nhau, công thức trong hình không thể hiện điều này.
        Ngoài ra có thể thấy khi ta đứng xa thì lượng ánh sáng nhận được giảm nhưng khi ta tăng diện tích bề mặt thì lượng ánh sáng thu được từ bề mặt lại tăng.

## Kết luận
**Image Intensity được tính dựa trên tổng lượng ánh sáng mà 1 điểm phát ra mà mắt ta/len camera thu được. Tuy nhiên nó không phải tất cả, bản chất mắt/camera là 1 thấu kính phức tạp, ánh sáng đi vào có vai trò là tia tới của thấu kính, tia đi ra sẽ phải hội tụ ở đâu đó trên 1 màn chắn nào đó (giác mạc/image plane). Độ sáng trên màn chắn của điểm trên màn chắn mới là Image Intensity. Bài sau ta sẽ nói về điều này.**