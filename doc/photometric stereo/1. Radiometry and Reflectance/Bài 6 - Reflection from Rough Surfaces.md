# Reflection from Rough Surfaces (bề mặt gồ ghề) | Radiometry and Reflectance

## Mục tiêu và vấn đề

![Rough surfaces](images/6.%20Surface%20Roughness.png)

Bằng kiến thức các bài trước, giá trị của 1 pixel chính là tổng lượng ánh sáng do 1 vùng diện tích nào đó của surface phát ra.

Tuy nhiên có những surface gồ ghề, những lúc thế này pixel thật khó để thể hiện rằng bề mặt mà nó đang phản chiếu là smooth irrespective (trơn tru). Thực tế hầu như bề mặt của vật thể luôn có 1 độ gồ ghề nào đó.

=> Tìm cách mô hình hóa roughness của 1 surface và xem nó làm thay đổi reflectance model thế nào.

## Modeling surface roughness - Micro-facet structure

![Model surface roughness](images/6.%20Model%20surface%20roughness.png)

1 trong những mô hình đơn giản nhất để mô hình hóa roughness của bề mặt là micro - facenet

Hãy tưởng tượng, bề mặt được tạo bởi các hạt rất nhỏ - facet.

Bản thân bề mặt sẽ có 1 mean orientation (Pháp tuyến trung bình, là đường vuông góc với bề mặt tại tâm). Các facet sẽ có pháp tuyến riêng và tạo với pháp tuyến n 1 góc alpha như hình vẽ.

Bằng cách định nghĩa như vật, ta có thể mô tả roughness của surface bằng cách sử dụng 1 qui luật phân bổ (distribution) nào đó.
    Ví dụ quy luật phân bổ gaussian => Gaussian Micro-Faccet model
    Ta tính được xác xuất phân bổ các facet có alpha từ 0 đến 90.
    Xích ma/độ lệch chuẩn trong công thức càng lớn thì bề mặt càng gồ ghề (độ lệch chuẩn bằng 0 thì là mặt phẳng hoàn hảo)
    
    ![standard devitation](images/6.%20standard%20devitation.png)

Kết hợp model này với lambertian model (body reflection) và specular model (specular reflection) ta sẽ có câu hỏi "what's the aggregate reflectance" (hệ số phản xạ tổng hợp) của toàn bộ lô mà bạn đang xem là gì

## Specular reflection from Rough Surface (sự phản xạ gương của 1 bề mặt gồ ghề)

Torrance-Sparrow BRDF model: Mỗi facet là 1 perfect mirror, mỗi facnet có thể có 1 vector pháp tuyến riêng điều đó tạo nên surface roughness. Khi bạn tính toán BRDF, bạn sẽ tính nó trên 1 tập hợp các facenet

![Specular Reflection](images/6.%20Specular%20Reflection.png)

    ps: độ phản chiếu của từng facenet. mặc dù là perfect mirror nhưng không có nghĩa là 100% ánh sáng nó nhận được sẽ bị phản chiếu. Điều này phụ thuộc vào chất liệu vật liệu.
    p(alpha, xích ma): là microfacnet ở trên tức là độ nhám
    G: được gọi là geometric attenuation factor (hệ số suy giảm hình học). Nó tính đến các hiệu ứng hình học khác nhau xảy ra khi bạn có 1 surface không phẳng

Tưởng tượng bạn có 1 mặt phẳng hình V như hình vẽ (2 facnet). Do đó khi ánh sáng đến từ 1 phía, có thể xuất hiện bóng (sự chiếu sáng - đường màu trắng và sự phản xạ của các tia chiếu - màu đỏ. Khi hướng quan sát là tai màu đỏ, ta sẽ không thấy được phần không phản xạ ánh sáng) => 1 facnet có thể che khuất khả năng hiển thị của facnet khác và hàm G này đề cập đến điều đó.

1. Khi xích ma thay đổi
    ![Specular Reflection from Rough Surface](images/6.%20Specular%20Reflection%20from%20Rough%20Surface.png)
    
    Nhắc lại: trong phản xạ gương ánh sáng chỉ có thể phản xạ theo 1 hướng, còn phản xạ body thì ánh sáng bị tán xạ theo nhiều hướng. Do đó khi view quan sát tại 1 hướng, họ chỉ nhìn thấy duy nhất 1 điểm sáng của vật thể do chỉ có 1 điểm sáng có tia phản xạ trùng phương với góc view.

    Khi xích ma = 0, bề mặt là mặt gương tuyệt đối do đó ta chỉ nhìn thấy 1 điểm sáng như hình
    Khi xích ma càng lớn hơn 0, bề mặt càng gồ ghề thì càng có nhiều điểm có tia phản xạ chiếu đến ống kính của camera dẫn thấy ta không chỉ nhìn thấy 1 điểm sáng nữa. Ngoài ra điểm sáng nhất trong trường hợp này thường không phải điểm p mà sẽ là 1 điểm khác gần đó.

2. Ví dụ
    ![](images/6.%20Specular%20Reflection%20from%20Rough%20Surface%201.png)

    Ví dụ dễ nhìn hơn là hình trên. Đó là 1 quả cầu và ta tăng độ gồ ghề/nhám của vật thì nó sẽ càng ngày càng mờ và ta càng khó để đoán được hướng của nguồn sáng từ quả cầu mờ này vì độ thô giáp. Mà đây là phản xạ gương chứ không phải là phản xạ body, không có sự phản chiếu lambertian nào xảy ra cả.

## Reflection from Rough Surfaces | Radiometry and Reflectance

Tương tự như phản xạ gương, nếu bề mặt lambertian (body reflectance) mà gồ ghề thì sao ?
1 pixel là tập hợp các facnet các lambertian hướng khác nhau.
Ta rút ra được công thức BRDF trong trường hợp này (không giải thích). Hiển nhiên xích ma mô tả độ gồ ghề của bề mặt và khi xích ma = 0, bề mặt trở thành perfect lambertian.

1. Khi xích ma thay đổi
    ![Body Reflection from Rough Surfaces](images/6.%20Body%20Reflection%20from%20Rough%20Surface%201.png)
    Khi bạn tăng xích ma sẽ thấy quả cầu càng phẳng và mịn hơn, độ sáng nhất của quả cầu sau không bằng độ sáng nhất của quả cầu ban đầu. => Hiệu ứng thú vị

2. Ví dụ thực tế hơn
    ![](images/6.%20Body%20Reflection%20from%20Rough%20Surface%202.png)
    Hiệu ứng này càng rõ rết nếu hướng nhìn và hướng nguồn sáng giống nhau (như cầm đèn pin soi)
    Khi xích ma bằng 0, bề mặt không gồ ghề và quả cầu là lambert hoàn hảo. Do 2 hướng giống nhau nên phần ở giữa sáng hơn những phần ở rìa.
    Tuy nhiên khi tăng xích ma lên nó càng ngày càng phẳng hơn và trong trường hợp xích ma = 0.3 thì quả cầu giống cái đĩa phẳng và đó cũng chính là hiện tượng quan sát được khi trăng tròn (trung thu, 15/16 hằng tháng)