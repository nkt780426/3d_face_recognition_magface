# Interreflections (phản xạ lẫn nhau)

Có 1 vấn đề mà ta đã gạt đi ngay từ đầu là interreflections (phản ánh)

![](images/14.%20Interreflection.png)

Giả sử có 1 vật thể như trên hình, nó là cái cốc và đang chụp. Phương pháp photometric stereo giả sử rằng 1 điểm (màu đen trên hình) chỉ nhận ánh sáng từ nguồn. Nhưng thực tế 1 điểm có thể nhận ánh sáng từ các điểm xung quanh nó do phản xạ, không nhất thiết là từ vật thể. Đây là vấn đề mà ta đã gạt đi trong kỹ thuật tái tạo 3D như photometric stereo vì không có cách giải quyết do ngay từ đầu ta đã không biết về hình dạng vật, chất liệu cũng như lượng ánh sáng mà các điểm khác phản xạ vào điểm quan sát.

Kết quả khi sử dụng phương pháp photometric stereo với vật thể lõm xuống như cái cốc, nó sẽ không còn chính xác vì hiệu suất phản chiếu (albedo) là rất lớn và ảnh hưởng đến phương pháp.

Ngoài ra độ nghiêng của bề mặt so với nguồn cũng bị đánh giá thấp hơn so với thực tế => Ảnh tái tạo các vật thể lõm sẽ bị nông hơn so với thực tế.

# Giải pháp: 

![](images/14.%20Solution.png)

Giả sử viền ngoài cùng là hình dạng chiếc cốc lambert muốn khôi phục. Khi áp dụng phương pháp photometric stereo, do interrèlections (phản xạ lẫn nhau) nên lúc tại tạo, ta tái tạo ra đường màu đỏ nông hơn so với đường gốc.

Thực tế ta phải biết cả 2 đường đó do ta có quyền can thiệp vào vật thể. Và khi có được 2 đường trên, ta sẽ dựa vào đường màu đỏ để tính toán ngược lại về mức độ phản xạ (interrèlections) mà mỗi điểm sẽ nhận được và từ đó điều chỉnh cường độ do mỗi nguòn sáng tạo ra rồi tính toán lại kết quả photometric stereo. Tiếp tục lặp lại quá trình này bạn sẽ thu được kết quả hội tụ đến gần vật thể nhất.

Tuy nhiên cách tái tạo này rất khó vì đòi hỏi ta phải biết được hình dạng thực sự của vật thể. Và còn phức tạp nữa, lặp bao h mới xong.