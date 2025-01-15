# Shape from Normals
Ở các bài trước, các hình ảnh 3D mà bạn nhìn thấy chỉ là tượng trưng. Mọi thứ mới chỉ dừng ở bước tính ra normal map. Bài này sẽ tìm hiểu cách dựng 1 hình ảnh 3D từ normal map thu được.

**Phương pháp này chỉ thực hiện được khi bề mặt liên tục, còn bề mặt gián đoạn (như giá trị tuyệt đối, bậc thang đột ngột đi xuống, ...) nếu bạn không biết độ sâu z mà nó đi xuống/lên thì sẽ không thể tái tạo hoàn toàn đối tượng.**

## Tổng quan

![](images/13.%20Generation.png)

Giả sử normal map bên trái tương đương với đối tượng ở phía bên phải. 1 đối tượng ngoài trục x, y mà ta nhìn thấy trong ảnh thì còn phải có trục z. Thứ ta cần là tái tọa được trục z từ cái normal map này.

Ta đã có p và q và nó chính là đạo hàm của z theo x và y đã học (xem hình trên). Đây là cách biến từ depth map (z) trở thành normal map. Tuy nhiên thứ ta cần là điều ngược lại, làm sao từ normal map tính ra được depth map (z) của vật thể => Sử dụng vi phân nguyên hàm lại.

![](images/13.%20Generation%202.png)

Tại 1 điểm (x0,y0) trên bề mặt ta coi nó là mốc (z(x0,y0)=0), độ sâu của mọi điểm bề mặt sẽ được tính toán theo mốc này tạo thành z.map.

Để tính độ sâu của các điểm còn lại, ta chỉ cần lấy tích phân từ điểm đó đến z0 là tính ra được z của nó.
z(x,y) được tổng hợp bởi tất cả các điểm nằm giữa nó và z(x0,y0). Nếu bạn tính toán đúng giá trị p,q thì bạn sẽ kết thúc tại đúng điểm z(x,y)

Từ những điều trên ta rút ra thuật toán tính z.map của vật thể từ normal map như sau.

![](images/13.%20Algorithm.png)

1. Chọn mốc z(0,0) = 0

2. Tính tích phân các điểm còn lại đến z(0,0) từ đó rút ra được giá trị z của chúng (tính theo chiều dọc rồi từ 1 cột tính được bắt đầu for tính tích phân các điểm chiều ngang)

## Noise Sensitivity of Computer Shape

Thật không may, việc tính toán không diễn ra thuận lợi. Vì ảnh có thể bị noise

![](images/13.%20Noise.png)

Như bạn có thể thấy trên hình, có rất nhiều điểm trên bề mặt phải cùng có vector pháp tuyến nhưng khi ta tính toán ra thì lại xiên xẹo. Điều này là do noise, có thể là do noise trong image intensity mà ta đo được (nhiễu máy ảnh), có thể là do sai lệch về các giá trị mà ta đã lấy nó chưa hoàn toàn chính xác, sai số lớn, ...

Nếu bạn cứ tiếp tục xử lý mà bỏ qua noise, bạn sẽ gặp trường hợp khi tính tích phân theo nhiều hướng sẽ đến nhiều điểm khác nhau chứ không hội tụ nữa.

1. Cách giải quyết 1: Tính toán depth map sử dụng các path khác nhau và thu được nhiều kết quả trong cùng 1 pixel. Tính trung bình các giá trị này và hi vọng giá trị này sẽ loại bỏ 1 số nhiễu

2. Cách giải quyết 2: sử dựng phương pháp Least Squares.
    ![](images/13.%20Least%20Square.png)
    Mục tiêu của cách này là giảm thiểu sai số của tham số p, q đo được.
    Sử dụng 1 hàm thước đo sai số D. Sau đó tính toán các path và tìm giá trị z làm cho D nhỏ nhất. 
    ...(xem lại youtube đoạn này) ....
    ![](images/13.%20Fourier.png)

    ![](images/13.%20Fourier%202.png)

## Kết quả

1. Chai nước
    ![](images/13.%20Bolder.png)
    Hình thứ 2 là deep map, cái nào càng gần camera hơn thì sáng hơn. Hình trên chỉ là tượng trưng, rất khó để kiểm tra chất lượng của deep map. Vì vậy 1 cách thường dùng là lấy depth map và đặt 1 vài BRDF lên nó và hiển thị nó bằng computer graphic sử dụng thêm 1 số nguồn sáng để có thể visualize được các gợn sóng/chi tiết của bề mặt.

    Có thể thấy kết quả sau khi thu được ở hình 3 rất chính xác.

2. Mặt nạ
    ![](images/13.%20Mark.png)

3. Fish toy
    ![](images/13.%20Fish.png)

    Vật thể trên là đồ chơi hình con cá và được làm bằng nhiều vật liệu khác nhau có hoa văn chi tiết phức tạp và có thể thấy 1 loạt các quả cầu được dùng để hiệu chỉnh kết quả.