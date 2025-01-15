# Gradient Space and Reflectance Map 
## Suface gradient and Normal

1. Công thức vector pháp tuyến n của mọi điểm trên bề mặt.
    Surface gradiant là cách dễ dàng để biểu diễn surface orientation.

    ![](images/9.%20Surface%20gradient%20and%20Normal.png)

    Trong gradient space. Hàm bề mặt sẽ được biểu diễn thành z = f(x,y). Trong đó z là độ sâu (depth) của bề mặt.
    Surface gradient được định nghĩa là đạo hàm của z đỗi với x và y => Được tạo độ (p,q) như trên hình => pq được gọi là gradient của surface trong không gian này.
    Surface normal được biểu diễn trong không gian này có tọa độ: N = (p, q, 1) như trên hình. Mà p, q lại phụ thuộc vào x, y nên công thức này là công thức surface normal chung mọi điểm trên bề mặt.

    Hiển nhiên chúng ta muốn đưa N về vector đơn vị. Như hình trên, ta thu được công thức chung của vector pháp tuyến đơn vị n của mọi điểm trên surface về mặt toán học và nó chỉ phụ thuộc vào 2 tham số p và q.

2. Sự phụ thuộc của vector pháp tuyến n với giá trị p,q
    ![](images/9.%20Surface%20gradient%20and%20Normal.png)

    Giả sử trong tọa độ Oxyz có vector pháp tuyến đơn vị n (unit normal).
    Dựng mặt phẳng (plane) z=1 song song với mặt phẳng Oxy và ta sẽ thu được trực p và q trên mặt phẳng này. Nó là các đường thẳng song song với Ox, Oy.
    Mặt phẳng z=1 chính là pq space hay gradient space.
    Để thu được tạo độ của vecto đơn vị n trên gradient space. Ta kéo dài vector đơn vị n cặt mặt phẳng gradient tại điểm N có tọa độ N(p, q, 1) là normal vector như trên hình
    **Mỗi điểm trong không gian gradient chỉ có duy nhất 1 vector n có hướng thỏa mãn trong không gian 3D.**

    Hiển nhiên ngoài surface normal, ta có thể miêu tả hướng của nguồn sáng, ... Công thức cũng sẽ tương tự.(xem hình về vector s).

## Reflectance Map

**Ta đã nói rất nhiều việc từ Image Intensity và các điểu kiện đã biết ta có thể tái tạo 3D. Reflectance Map là cách để biểu diễ n tất cả những cái đã biết đó về 1 thứ duy nhất.**

1. Reflectance map
   
    Sau khi hiểu về cách biểu diễn các vector trong hệ tọa độ gradient space, ta nâng cao hơn tý nữa được khái niệm Reflectance Map

    ![](images/9.%20Refectance%20Map.png)

    Giả sử rằng chúng ta đã có các thuộc tính phản xạ của bề mặt (reflectance properties) như BRDF của bề mặt, các thông tin về nguồn sáng (single source, có 1 độ sáng và khoảng cách nhất định đến bề mặt và hướng của nó trong gradient space)

    Reflectance Map cho ta biết rằng đối với 1 hướng của nguồn và surface reflectance model nhất định. Ta nhân với Reflectance map sẽ thu được Image Intensity tại điểm (x, y). **Hiểu đơn giản nếu ta biết được hướng của nguồn sáng đối với 1 điểm thuộc scene và reflectance map của nó. Ta có thể tính được image ỉntensity của điểm đó.**

    Nhưng mà bài toán của chúng ta là ngược lại, từ image intensity, ta muốn biết surface normal của điểm đó trên scene. Hãy xem xét nó với các mô hình phản xạ sau.

2. Reflectance Map của Lambertian Surface
    
    ![](images/9.%20Lambertian%20surface.png)

    Lambert surface rất phổ biến trong thực tế, điểm đặc biệt của nó là độ sáng bề mặt không phụ thuộc vào hướng mà ta quan sát (nhìn hình có thể thấy các tia phản xạ được phản xạ đều theo mọi hướng) nó chỉ phụ thuộc vào hướng chiếu sáng s của nguồn.

    ![](images/9.%20Reflectance%20Map%20Lambertian.png)

    Giả sử ta đã biết hướng s của nguồn => Intensity đo được trên 1 lambert surface sẽ có công thức như góc trái của hình.
        albedo: reflectance của bề mặt lambert. Albedo = 0, bề mặt màu đen, albedo = 1 thì bề mặt màu trắng tượng trưng cho tất cả ánh sáng được phản chiếu lại
        c: là gain của máy ảnh (tóm lại là thông số của camera như độ dài ống kính, đường kính, ... Tất cả tổng hợp thanh 1 tham số c)
        l = J/r^2
    Do ta đã biết mọi điều kiện chiếu sáng, nên ta có thể chọn 1 môi trường sao cho c * p / k = 1. Khi đó công thức được thu gọn thành I = n * s (trong đó n và s là vector đơn vị)

    ![](images/9.%20Reflectance%20Map%20Lambertian%202.png)

    Đến đây ta biết được rằng R(p,q) = I = n * s => Khi 1 điểm càng sáng thì giá trị R càng lớn. 
    
    Một câu hỏi đặt ra là trong hệ tọa độ trên có những điểm nào có độ sáng bằng nhau (các điểm có độ sáng bằng nhau sẽ được vẽ thành 1 đường viền ở phần tiếp theo)

3. Reflectance Map: Iso-Brightness Contours (đường viền)

    ![](images/9.%20iso-brightness%20contours.png)

    Để trả lời cho câu hỏi tập hợp các điểm trên gradient space (mỗi điểm đại diện bởi 1 hướng của nguồn sáng).
    Chúng ta biết rằng đối với bề mặt lambert, các surface normal cùng tạo với s 1 góc giống nhau, tát cả những thứ này sẽ tạo ra cùng 1 giá trị độ sáng. Bây giờ ta chỉ cần kéo dài cái hình nón này ra cho nó cắt với z = 1 sẽ thu được iso-brightness contours.

    ![](images/9.%20Reflectance%20Map%20Lambertian%203.png)

    Kết quả ta thu được như trên hình, các điểm thuộc cùng 1 iso-brightness contour sẽ có cùng độ sáng.

    ![](images/9.%20Reflectance%20Map%20Lambertian%203.png)

    Điểm có R(p,q) = 1 hay điểm sáng nhất thì surface normal map cùng phương với vector s hay cosin theta = 0 và khi giá trị này giảm, ta có thể thấy các đường iso-brightness càng đi xa  và khi I =0 ta có tử bằng 0 và có được đường trên kia.

## Shape from normal

Quay lại vấn đề, khôi phục ảnh 3D từ Image Intensity và Reflectance Map (nguồn sáng s và các đặc tính phản xạ của bề mặt) thế nào. Liệu có thể khôi phục 3D từ 1 ảnh duy nhất không ?

![](images/9.%20Shape%20from%20normal.png)

Hay đơn giản vấn đề, ta có thể ước tính surface gradient (hay p, q) ở mỗi pixel ? => Câu trả lời là 0.
- Hãy nhìn vào hình trên, xét 1 điểm I nào đó trên image. Giả sử nó ánh xạ vào được reflectance map. Tuy nhiên trong reflectance map, các điểm có độ sáng giống nhau có thể nằm trên cùng 1 iso-brighness => Không thể xác định chính xác pixel nó là điểm nào trong reflectance map => Không thể xác định được (p, q)
- Photometric stereo là phương pháp sử dụng nhiều ảnh với nhiều nguồn sáng ở vị trí khác nhau để giải quyết vấn đề này và tạo ra surface normal duy nhất tương ứng với mỗi pixel của ảnh.