# Dichromatic Model (mô hình phản xạ lưỡng sắc) | Radiometry and Reflectance
## Mục tiêu và vấn đề
Ở 6 bài trước, ta đã thảo luận về 2 cơ chế phản xạ cơ bản là surface reflection (specular reflection) và body reflection. Điều chúng ta chưa bao giờ thảo luận về 2 kiểu phản xạ này là màu sắc của nguồn sáng.

Mối quan hệ giữa màu sắc của ánh sáng nguồn chiếu tới và tia phản xạ ?

## Color reflectance model

1. Tổng quan màu sắc của ánh sáng nguồn với 2 loại phản xạ
    ![Color Reflectance model](images/7.%20Color%20Reflectance%20Model.png)
    
    Đối với phản xạ gương, hầu hết các vật liệu đều cho ra tia phản xạ có màu sắc giống với ánh sáng từ nguồn chiếu tới
    
    Tuy nhiên với phản xạ body thì lại khác, do ánh sáng từ nguồn chiếu tới đi vào 1 phần bề mặt vật liệu. Mà vật liệu có thể có màu sắc của riêng nó, những tia sáng này sẽ hấp thụ 1 phần màu sắc của nó rồi mới đi ra ngoài, phản xạ body là vậy. Trong toán học gọi màu sắc của tia phản xạ bằng tích chập của màu sắc ánh sáng tới và màu của vật thể (bao nhiêu phần trăm thì chịu)

    => Kết luận: Khi ta nhìn vào màu sắc của 1 pixel trong ảnh chúng ta có thể hiểu rằng đố là sự kết hợp tuyến tính giữa body reflect và color of surface. Đọc phần 2

2. Dichromatic model (mô hình phản xạ lưỡng sắc)
    
    ![Dicromatic model](images/7.%20Dicroatic%20model.png)

    Nhìn vào hình có thể thấy 3 trục RGB đại diện cho RGB color space (3 chiều).

    Tia màu hồng đại diện cho màu của body reflection, màu vàng đại diện cho màu của surface/nguồn sáng

    Màu của pixel là hợp của 2 vector trên (tia màu) 
    => Mặt phẳng chứa 2 tia này gọi là Dichromation Plane (mặt phẳng lưỡng sắc), tất cả các điểm trên vật thể mà đo được nằm trong mặt phẳng này (các điểm đó phải cùng 1 vật liệu) thì sẽ cho ra mà sắc giống nhau và có cùng 1 giá trị pixel trong ảnh (cùng color).

3. Ví dụ trực quan

    ![](images/7.%20Color%20Reflectance.png)

    Trên hình là 1 quả cầu màu đỏ được chiếu sáng bởi ánh sáng có màu xanh lam.

    Khi ánh xạ tất cả vào các pixel trên image, bạn sẽ có sự phân bổ nhưng hình lập phương bên phải. Trong cái hình lập phương có 1 mặt phẳng đó là Dichromatic model và các vector color reflectance và color surface

    Trong hình trên, Dichromatic model bị cắt góc nhỏ 1 phần (cliping). Về đơn giản là hình ảnh bị bão hòa (saturated) ở 1 kênh màu nhưng không bão hòa ở kênh màu còn lại.

    Đặc biệt trong hình trên, sự phân bổ màu sắc của vật thể là skewed Thực
        Khái niệm skewed T (Skewed T-distribution) liên quan đến mô hình Dichromatic trong các lĩnh vực như xử lý hình ảnh, đồ họa máy tính hoặc phân tích dữ liệu có thể mô tả một dạng phân phối T (T-distribution) bị lệch (skewed)
        
        Nghĩa là không đối xứng quanh giá trị trung tâm. Skewed T thường xuất hiện trong những trường hợp dữ liệu không tuân theo phân phối chuẩn hoặc có nhiều giá trị cực đoan về một phía hơn phía còn lại.

        Trong xử lý hình ảnh và các mô hình màu sắc (như Dichromatic model), skewed T có thể dùng để mô tả phân phối của các đặc điểm như ánh sáng, màu sắc, hay độ sáng khi các yếu tố này không được phân bố đồng đều do ảnh hưởng của ánh sáng bất đối xứng hoặc các điều kiện môi trường khác.

        Tóm lại, "skewed T" liên quan đến một dạng phân phối có đặc tính bất đối xứng và thường được áp dụng trong các mô hình toán học khi dữ liệu không tuân theo các phân phối chuẩn đối xứng.

4. Tách body and surface reflection từ 1 ảnh

    ![](images/7.%20Separating%20body%20and%20surface%20reflection.png)

    Từ 1 bức ảnh chụp duy nhất bên trái ta có thể tách các body reflection component và surface reflection component của vật thể.

    Đầu tiên tách 3 chiếc cốc ra. Sau đó với mỗi chiếu cốc thực hiện ánh xạ từng pixel vào trong RGB color space hoặc color space nào đó bạn sẽ có được sự phân bố như hình bên phải và nó tuân theo quy luật phân bố Skewed T.

    Nhìn vào sự phân bổ các pixel (mấy cái dấu chấm), ta sẽ vẽ được mặt phẳng Dichromatic model và xác định được 2 vector màu hồng (body reflect) và vàng (specular reflect).

    Sau khi xác định được 2 vector trên, ta lần lượt chiếu các pixel xuống 2 vector này để tìm tọa độ của chúng => Tách được pixel thành 2 thành phần body reflectance và specular reflectance.
    
    Kết quả thu được như sau

    ![](images/7.%20Separating%20body%20and%20surface%20reflection%201.png)

    **Điều này rất hữu ích vì specular reflectance thường gây phiền toái trong computer vison thì nó thường float (nổi) on the top of the object và di chuyển theo nguồn sáng trong khi body reflectance tiét lộ cấu trúc 3 chiều của vật thể**