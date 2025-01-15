# Calibration Based Photometric Stereo | Photometric stereo

## Mục tiêu và vấn đề
Bài trước ta thực hiện phương pháp Photometric stereo với bề mặt lambertian. Hình ảnh estimated albedo chính là hình ảnh 3D của vật thể mà ta thu được (ảnh này có đặc điểm là thể hiện rõ chi tiết của vật thể mà không phụ thuộc vào nguồn sáng)

Để làm được điều này ta phải đảm bảo 1 số điều sau
- Phải thu thập ít nhất 3 ảnh với 3 nguồn sáng khác nhau, và hướng của 3 nguồn sáng phải độc lập tuyến tính với nhau.
- Mọi điểm trong hình phải được nhìn thấy bởi nguồn sáng, nếu không được phải thu thập thêm nhiều hình khác chứa điẻm nó không thấy.

=> Vấn đề: Không phải mọi điểm trên vật thể đều có BRDF là lambertian mà có thể lẫn với specular/surface reflectance dẫn đến sai lệch trong tính toán albedo và sai giá trị của normal vector. Điều này thường xảy ra với vật thể nhựa hoặc vật thể được làm bởi vật liệu khác nhau được sơn phủ lên.

Giải pháp: Mặc dù có rất nhiều loại nhựa, nhưng ta sẽ chỉ định rằng vật liệu được làm từ 1 loại nhựa duy nhất. Từ đó ta sẽ biết được BRDF của vật thể và dùng nó làm đối tượng hiệu chỉnh (calibration object) để áp dụng phương pháp photometric stereo.

## Calibration based photometric stereo

![](images/12.%20Calibration%20based%20photometric%20stereo%201.png)

Ở hình scene, ta có thể thấy vật liệu được sơn bởi 1 cái gì đó và có specular reflection luôn, không phải hoàn toàn là lambertian và chúng ta không biết BRDF của vật liệu này tại từng điểm trông như thế nào.

Ta sẽ sử dụng 1 vật liệu có BRDF tương tự => Đó là quả cầu, được gọi là calibration sphere (quả cầu hiệu chỉnh).

Ở quả cầu hiệu chỉnh, ta sơn cùng 1 loại sơn lên nó. Vì vậy ta biết rằng 2 quả cầu này là cùng 1 vật liệu có các đặc tính phản xạ giống nhau. Chúng ta sẽ chiếu sáng 2 vật thể này bằng cùng 1 nguồn sáng.

Nhìn vào 2 vector pháp tuyến của 2 vật thể trên hình, ta có thể thấy intensity của 2 điểm trên quả cầu giống với intensity của chai nước

Ta thực hiện thay đổi nguồn sáng

![](images/12.%20Calibration%20based%20photometric%20stereo%202.png)

![](images/12.%20Calibration%20based%20photometric%20stereo%203.png)

Có thể thấy khi bạn thay đổi nguồn sáng thì 2 điểm đó trong mọi điều kiện sẽ có cùng Image Intensity. Điều này được gọi là sự nhất quán trong định hướng (orentation consistency)

**Khi nhìn vào 1 surface normal của vật thể gốc (scene) và nhìn vào surface normal của vật thể celebration (quả cầu). Khi 2 surface normal này bằng nhau thì các image intensity mà ta thu được từ các ảnh sẽ đồng bộ với nhau giữa 2 ảnh. Đây chính là cách mà ta giải quyết vấn đề.**

## Calibration proceduce

![](images/12.%20Step%201,2.png)

1. Chụp 1 loạt các tấm ảnh từ vật thể hiệu chỉnh (celebration object)
    **Nếu bạn chụp k ảnh thì phải đảm bảo vật thể của bạn cũng phải chụp n ảnh và có nguồn sáng giống với k ảnh này.**
    Bây giờ mỗi điểm trên images cung cấp K image intensity tương ứng với k nguồn sáng/ảnh chụp.

2. Lấy 1 image bất kỳ và vẽ ra đường viền của vật thể. Trong hình trên là vòng tròn vật thể. Thứ này giúp chúng ta xác định được mục tiêu tìm normal map. Sau đó thực hiện tìm normal thôi.

![](images/12.%20Step%203.png)

3. Tạo lockup table như trên hình

![](images/12.%20Step%204,5.png)

4. Sau khi có được bảng lockup của p,q đối với vật thể hiệu chỉnh, ta thực hiện chụp k ảnh tương ứng với k ảnh của vật thể hiệu chỉnh. Từ các intensity tupple thu được ta tìm ngay ra được normal map của vật thể