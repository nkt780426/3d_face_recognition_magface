# Lambertian Case

## Photometric Lambert case với góc nhìn đại số.

Ở bài trước ta đã nhìn thấy góc nhìn hình học của phương pháp photometric, bài này ta sẽ nhìn vào góc nhìn đại số của phương pháp (là công thức I = n * s)

![](images/11.%20Lambertian%20case.png)

Tổng hợp 3 phương trình trên thành 1 ma trận ta được kết quả như hình trên. (p là albedo - hiệu suất phản chiếu)
2 ma trận nguồn sáng và image intensity là 2 ma trận đã biết, thứ ta không biết là albedo và normal map n. Albedo là hiệu suất phản chiếu bề mặt, giả sử rằng nó khác nhau tại mỗi điểm trên bề mặt (ví dụ như chiếc cốc giấy có họa tiết),

![](images/11.%20Lambertian%20case%202.png)

Normal vector bằng albedo/pi*n. Từ đó ta thay vào công thức được I=SN và dễ dàng giải được bằng cách tìm ma trận nghịch đảo của S và nhân 2 về như trên hình. Sau khi có được N ta thu được n và albedo.

## What's doesn't work

Cách tiếp cận này không hoạt động khi nào ?

![](images/11.%20Photometric%20Stereo%20not%20work.png)

- Nếu ma trân S là mà trận không khả nghịch (not invertible) tức là 3 nguồn sáng s cùng nằm trên 1 mặt phẳng (do đó 1 s là tổng hợp của 2 s còn lại - tổ hợp tuyến tính, không phải là độc lập tuyến tính toàn bộ)

- Bạn phải đảm bảo điều này không xảy ra khi sử dụng phương pháp photometric stereo.

Trong trường hợp muốn tạo photometric stereo ngoài trời thì sao. Mặt trời di chuyển trên 1 mặt phằng, tại sao không thiết lập 1 camera nhìn vào 1 building chẳng hạn và mặt trời sẽ là nguồn sáng di chuyển, ta có thể thực hiện tính toán surface normal vào cuối ngày. Quả thực điều đó đúng, tuy nhiên cũng có 1 số ngày tồi tệ với phương pháp photometric stereo nếu làm theo cách này và đó là ma trận S khả nghịch. 

![](images/11.%20Bad%20days.png)

## More sources than minimum needed

![](images/11.%20More%20sources%20than%20minimum%20needed.png)

Nếu bạn có nhiều ảnh với nhiều nguồn sáng khác nhau hơn so với 3 ảnh tối thiểu thì phương pháp này hoạt động như thế nào /
Hiển nhiên nhiều ảnh hơn thì kết quả thu được phải mạnh mẽ hơn. Lúc này S không còn là ma trân vuông nữa và không thể nghịch đảo và khả nghịch. => Sử dụng phương pháp Least Square (bình phương tối thiểu)

Nhân cả 2 vế với ma trận ST => Làm tiếp như trên hình ta thu được ma trận N

## Effecttive point source for multiple sources

Đối với bề mặt Lambertian, multiple source khi xuất hiện cùng 1 lúc trên scene kể cả extending source cũng có thể được đưa về nguồn sáng điểm. Điều này chỉ xảy ra khi làm việc với bề mặt lambert

![](images/11.%20Effective%20point%20source%20for%20multiple%20sources.png)

Theo hình trên ta có thể thấy I mà image thể hiện chính là 1 nguồn sáng điểm tổng hợp của 2 nguồn sáng do đó định lý trên đúng. Thậm chí với nguồn sáng mở rộng ta có thể quy mạp cũng ra kết quả tương tự => Ta thật sử có thể đưa multi source về 1 nguồn sáng điểm duy nhất.

Tuy nhiên bạn phải đảm bảo mọi điểm trên bề mặt lambert trong image phải nhìn thấy cả 2 nguồn sáng, không có vật cản nào.

## Result

1. Ví dụ 1

    ![](images/11.%20Result%201.png)

    Trên hình là 5 bức ảnh được chụp ở 5 điều kiện có nguồn sáng khác nhau (quả cầu có họa tiết - albedo). Chúng ta sử dụng phương pháp photometric stereo với 5 phương pháp trên. 

    Sử dụng 5 nguồn sáng là vì nếu sử dụng 3 nguồn sáng 3 ảnh thì có thể 1 điểm nào đó trên scene không thể được chiếu sáng bởi cả 5 nguồn sáng. 

    Kết quả thu được là normal vector (p, q) của vật thể tại mỗi điểm. Các giá trị này dùng để tính toán surface normal của vật thể, hình dưới bên trái chính là hình ảnh normal map của vật thể.

    **Từ bản đồ normal map, ta thu được kết quả vật thể 3D như ở hình bên phải. Đặc điểm của hình này là nó không bị ảnh hưởng bởi nguồn sáng (albedo giờ thành như nhau), thể hiện rõ cấu trúc của vật thể, từ đó cung cấp nhiều chi tiết cho quá trình training hơn.**

2. Ví dụ 2

    ![](images/11.%20Result%202.png)

    Là 1 chiếc mặt nạ được sơn, do nó là bề mặt khá nông (fairly shallow surface - tức là không có nhiều chi tiết, đa số là nổi) nên chỉ cần 3 ảnh chụp với 3 nguồn sáng khác nhau là đủ.

    Sử dụng phương pháp photometric stereo ta thu được hình ảnh của vector pháp tuyến và hình dạng vật thể 3D (albedo map). Có thể thấy các ảnh input là có nhiều biến thể về độ sáng, độ bóng trong mỗi hình ảnh này nhưng albedo ơ ảnh xử lý sau cùng không đổi ở mọi điểm. Nghĩa là vật liệu này được làm bằng 1 vật liệu.

3. Ví dụ 3

![](images/11.%20Result%203.png)

Khi nói về shape của vật thể, bạn có thể sử dụng surface normal được tính toán với từng kênh màu trong ảnh, hoặc bạn có thể kết hợp chúng theo cách nào đó để giảm noise.

Ví dụ này về 1 con gấu bông với các chi tiết độ sâu phức tạp và được chụp 4 bức ảnh. Bằng việc sử dụng photometric stereo ta thu dược normal map của mỗi kênh màu và tái tạo nó được hình Estimated albedo như trên.

Khi nhìn vào hình estimated albedo được dựng từ albedo.map, các cạnh xung quanh vấn đề (ví dụ như đoạn giữa cánh tay và chân). Lý do là albedo quá mạnh so với thực tế khiến ta nnhầm lẫn. Thực tế là ta giả định rằng con gấu bông đang ở bề mặt lambert nhưng thực tế có 1 số chỗ như hình này không phải là bề mặt lambert hoàn hảo mà có thể lẫn specular/surface reflectance.

*Nếu muốn nhìn rõ hơn điểm nào có vấn đề thì xem lại youtube.*

Khi đã tính sai albedo thì normal map chắc chắn có vấn đề vì n là vector đơn vị được tính từ công thức N=q/pi*n/.

Hiển nhiên trong thực tế gặp rất nhiều trường hợp như thế này, rất nhiều trường hợp bạn không thể biết được BRDF của từng điểm trên vật thể. Để giải quyết nó đọc bài tiếp theo.