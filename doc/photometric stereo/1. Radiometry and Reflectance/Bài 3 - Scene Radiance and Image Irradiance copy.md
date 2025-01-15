# Scene Radiance (brightness of scene) and Image Irradiance (Thông lượng ánh sáng/đơn vị diện tích ở trong ảnh)

## Mục đích và vấn đề: 
Thiết lập mối quan hệ giữa brightness của 1 điểm trong scene và brightness tương ứng của điểm đó trong image - Image Intensity.
Các khái niệm:
- Scene Radiance: Khái niệm cuối bài 2, là 1 công thức để ta tính được tổng lượng ánh sáng phát ra từ bề mặt vật liệu đi vào thấu kín.
- Image Irradiance: 'thông lượng ánh sáng của của 1 vùng bề mặtk' đi vào 1 đơn vị diện tích trong image plane. Nó là 1 địa lượng để tính Image Intensity.

1. Vấn đề:

    ![Scene Radiance và Image Irradiance](images/3.%20Scene%20Radiance%20and%20Image%20Irradiance.png)
    
    L: scene radiance của điểm đó (đại lượng đặc trưng lượng ánh sáng mà điểm đó phát ra đến 1 hướng nào đó với 1 góc nào đó)

    E: image Irradiance. Các tia phản xạ sẽ được thầu kính camera thu lại và hội tụ trên image plane. Đại diện cho độ sáng của điểm trong ảnh.
    
    Mục tiêu là tìm quan hệ giữa E và L

2. Máy ảnh

    ![camera](images/3.%20camera.png)

    Len - thấu kính của cảm biến máy ảnh.
    f - không phải tiêu cự của cảm biến máy ảnh mà là khoảng cách của Image Plane và Lens
    Image Plane - Màn thu ảnh.

3. Mối liên hệ giữa Scene Radiance và Image Irradiance

    ![pixel](images/3.%20pixel.png)

    dAi chính là pixel, giá trị của pixel được đo bởi lượng ánh sáng mà nó nhận được từ vùng có diện tích dAs.
    
    Do dAs rất nhỏ nên có thể coi nó là 1 mặt phẳng và có vector pháp tuyến n.
    
    Tóm lại ta sẽ rút ra được 1 số phương trình từ cái hình trên và nó thể hiện mối quan hệ giữa E và L

    ![Công thức 1](images/3.%20Equation1.png)

    Bắt nguồn từ việc soild angles dwi=dws

    ![Công thức 2](images/3.%20Equation2.png)

    Bắt nguồn tự độ dài thấu kính d và góc dwL hợp bởi thấu kính và dAs

    Tất cả lượng ánh sáng mà thấu kính nhận được từ dAs đều sẽ được chiếu đến pixel dAi. Bằng công thức Scene Radiance ở bài 2 ta tính được lượng ánh sáng này và có công thức 3.

    ![Công thức 3](images/3.%20Equation3.png)

    ![Công thức 4](images/3.%20Equation4.png)

    Image Irradiance (lượng ánh sáng mà image plane nhận được trên 1 đơn vị diện tích, tương tự như Sene Irradiance)

    Từ các công thức trên ta rút ra được mối quan hệ giữa L và E

    ![L and E](images/3.%20L%20and%20E.png)

    ![](images/3.%20Scene%20Radiance%20and%20Image%20Irradiance%202.png)

    - d là đường kính thấu kính
    - f lá tiêu cự thấu kính
    - alpha là góc ánh sáng chiếu tới thấu kính

    Có vài điều quan trọng mà công thức trên nói lên
    - Image Irradiance tỷ lệ thuận với Scene Radiance.
    - Độ sáng của Image sẽ giảm dựa vào cách bạn đặt camera do góc alpha. Tuy nhiên đây chỉ là vấn đề của ống kính đơn, ống kính phức tạp sẽ giảm thiểu điều này.

4. Độ sáng của image có ảnh hưởng đến độ sâu của scene (scene depth là khoảng cách của camera và scene) ?

    ![scene depth](images/3.%20deep%20scene.png)

    Hiểu nôm na khi ta dịch chuyển máy ảnh ra xa, độ phát sáng bề mặt của vật thể có thay đổi ?
    Dĩ nhiên là không. Nhưng mà độ sáng mà camera nhận được để tái tạo từng pixel sẽ giảm, và máy tính làm sao để hiểu và tính chính xác được độ sáng của scene thông qua imgae ?

    Theo công thức trên, ta có thể thấy nó không liên hệ gì đến khoảng cách của scene và camera. Nghĩa là Image brightness không ảnh hưởng đến scene depth. Tại sao lại như vậy ? Hãy nhìn vào hình minh họa

    Do kích thước pixel không thay đổi nên khi dịch chuyển camera đi xa thì vùng diện tích mà scene tương ứng với pixel tăng lên dẫn đến lượng ánh sáng mà pixel nhận được tăng lên.

    Tuy nhiêu soild angle mỗi điểm trong scene đến camera sẽ giảm do đó lượng ánh sáng mà pixel thu được từ mỗi điểm trong scene sẽ giảm đi (scene radiance giảm). Điều này bù trừ đi sự tăng lên về diện tích phát sáng tương ứng với camera. Kết quả là độ sáng của ảnh không phụ thuộc vào scene depth.